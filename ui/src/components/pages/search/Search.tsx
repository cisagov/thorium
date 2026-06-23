import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Card, Col, Row, Stack } from 'react-bootstrap';
import DOMPurify from 'dompurify';
import AlertBanner from '@components/shared/alerts/AlertBanner';
import parse from 'html-react-parser';
import styled from 'styled-components';

// project imports
import { OmnibarMainSearch } from '../../shared/inputs/omnibar/Bars';
import { Clause } from '../../shared/inputs/omnibar/ClauseTypes';
import { TimeSelection, TimeSelectionToStrings, defaultTimeSelection } from '../../shared/inputs/omnibar/timepicker/utils';
import { useOmnibarUrlState } from '../../shared/inputs/omnibar/useOmnibarUrlState';
import {
  getGroupsFromClauses,
  getIndexesFromClauses,
  getLimitFromClauses,
  getSearchTextFromClauses,
} from '../../shared/inputs/omnibar/utils';
import EntityList from '@entities/browsing/EntityList';
import { search } from '@thorpi/search';
import { OmniClauseAndTimeToFilter } from '@utilities/search';
import { ElasticDoc, SearchFilters } from '@models/search';
import { scaling } from '@styles';

// get hash of a file from result ID
const getSha256 = (id: string) => {
  const splitID = id.split('-');
  if (splitID.length > 0) {
    return splitID[0];
  }
  return '';
};

// get group name from result ID
const getGroup = (id: string) => {
  const splitID = id.split('-');
  if (splitID.length > 1) {
    // get everything after the first element and join it back
    return splitID.slice(1).join('-');
  }
  return '';
};

// map a full index name given by Elastic to a simpler one
const mapFullIndexName = (fullIndexName: string) => {
  // TODO: matches based on the full name of the elastic index set
  //       in the Thorium config...not sure how to match that dynamically
  if (fullIndexName === 'thorium_sample_tags') {
    return 'Tags';
  } else if (fullIndexName === 'thorium_sample_results') {
    return 'Results';
  } else {
    return null;
  }
};

// replace kibana mark up tags w/ highlight html tag
const highlightResult = (result: string) => {
  const highlightStart = result.toString().replaceAll('@kibana-highlighted-field@', '<mark>');
  const highlightFinish = highlightStart.replaceAll('@/kibana-highlighted-field@', '</mark>');
  // we must sanitize the output that will be rendered as html
  const clean = DOMPurify.sanitize(highlightFinish, { ALLOWED_TAGS: ['mark'] });
  return parse(`${clean}`);
};

const Name = styled(Col)`
  white-space: pre-wrap;
  text-align: center;
  flex-wrap: wrap;
  word-break: break-all;
  min-width: 650px;
  color: var(--thorium-text);
`;

const Groups = styled(Col)`
  flex-wrap: wrap;
  text-align: center;
  min-width: 150px;
  color: var(--thorium-text);
  @media (max-width: ${scaling.lg}) {
    display: none !important;
  }
`;

const Index = styled(Col)`
  flex-wrap: wrap;
  text-align: center;
  min-width: 100px;
  color: var(--thorium-text);
  @media (max-width: ${scaling.xl}) {
    display: none !important;
  }
`;

const SearchResultsHeaders = () => {
  return (
    <Card className="panel">
      <Card.Body className="px-0">
        <Row>
          <Name>SHA256</Name>
          <Groups>Group</Groups>
          <Index>Index</Index>
        </Row>
      </Card.Body>
    </Card>
  );
};

interface SearchResultItemProps {
  result: ElasticDoc;
  idx: number;
}

const SearchResultItem: React.FC<SearchResultItemProps> = ({ result, idx }) => {
  return (
    <Card className="panel">
      <Row>
        {/* add common relative spacing for sha, group, and index*/}
        <Name>
          <Link to={`/file/${getSha256(result.id)}`}>{getSha256(result.id)}</Link>
        </Name>
        <Groups>{getGroup(result.id)}</Groups>
        <Index>{mapFullIndexName(result.index)}</Index>
        <hr />
      </Row>
      {result.highlight &&
        Object.keys(result.highlight).map(
          (key) =>
            key !== 'group' && (
              <Row key={`${getSha256(result.id)}_${idx}_${key}`}>
                <Col>
                  <span>
                    {key}: {highlightResult(String(result.highlight?.[key]))}
                  </span>
                </Col>
              </Row>
            ),
        )}
    </Card>
  );
};

// get repos using filters and and an optional cursor
const getSearchResults = async (
  query: string,
  clauses: Clause[],
  omniTime: TimeSelection,
  setSearchError: (error: string) => void,
  cursor: string | null,
): Promise<{ entitiesList: ElasticDoc[]; entitiesCursor: string | null }> => {
  if (query !== '') {
    const groups = getGroupsFromClauses(clauses);
    const indexes = getIndexesFromClauses(clauses);
    const limit = getLimitFromClauses(clauses, 25);
    const [end, start] = TimeSelectionToStrings(omniTime);

    // get files list from API
    const { entityList, entityCursor } = await search(query.trim(), setSearchError, indexes, groups, start, end, cursor, limit);
    return {
      entitiesList: entityList,
      entitiesCursor: entityCursor,
    };
  }
  return {
    entitiesList: [],
    entitiesCursor: null,
  };
};

const SearchBarContainer = styled.div`
  max-width: 1000px;
  width: 100%;
  display: flex;
  justify-content: center;
  position: relative;
  margin-bottom: 20px;
`;

// component containing search bar and related functionality
const Search = () => {
  // omnibar clauses + time live in the URL so searches are shareable/bookmarkable
  const {
    clauses: omnibarClauses,
    setClauses: setOmnibarClauses,
    time: omniTime,
    setTime: setOmniTime,
  } = useOmnibarUrlState({
    clauses: [],
    time: defaultTimeSelection(),
  });
  const [searching, setSearching] = useState<boolean>(true);
  const [filters, setFilters] = useState<SearchFilters>({ query: '' });
  // the id of the cursor for paging search results;
  const [searchError, setSearchError] = useState<string>('');
  const [debouncedQuery, setDebouncedQuery] = useState<string>('');

  // debounce omnibar changes into the search query + filters that drive EntityList; deriving
  // filters from the full clause+time state means changing group/index/limit/time alone (not just
  // the query text) also re-triggers the search
  useEffect(() => {
    const handleSetQuery = setTimeout(() => {
      const query = getSearchTextFromClauses(omnibarClauses);
      setDebouncedQuery(query);
      setSearchError('');
      setFilters({ ...OmniClauseAndTimeToFilter(omnibarClauses, omniTime), query });
    }, 500);
    return () => clearTimeout(handleSetQuery);
  }, [omnibarClauses, omniTime]);

  useEffect(() => {
    setSearching(false);
  }, []);

  return (
    <Stack>
      <div className="d-flex flex-row justify-content-center">
        <SearchBarContainer>
          <OmnibarMainSearch clauses={omnibarClauses} setClauses={setOmnibarClauses} time={omniTime} setTime={setOmniTime} />
        </SearchBarContainer>
      </div>
      {omnibarClauses.length > 0 && searchError === '' && (
        <Row>
          <EntityList
            type="Results"
            entityHeaders={<SearchResultsHeaders />}
            displayEntity={(result, idx) => <SearchResultItem result={result as ElasticDoc} idx={idx} />}
            filters={filters}
            fetchEntities={(_, cursor) => getSearchResults(debouncedQuery, omnibarClauses, omniTime, setSearchError, cursor)}
            setLoading={setSearching}
            loading={searching}
          />
        </Row>
      )}
      {searchError && omnibarClauses.length > 0 && <AlertBanner className="mt-1 mb-0">{searchError}</AlertBanner>}
    </Stack>
  );
};

export default Search;
