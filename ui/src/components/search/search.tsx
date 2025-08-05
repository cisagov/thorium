/* eslint-disable react-hooks/exhaustive-deps */
import React, { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Alert, Card, Col, Form, Row, Stack, Dropdown, DropdownButton } from 'react-bootstrap';
import DOMPurify from 'dompurify';
import parse from 'html-react-parser';

// project imports
import { BrowsingFilters, EntityList, IndexSelect } from '@components';
import { useAuth } from '@utilities';
import { search } from '@thorpi';
import { SearchFilters, ElasticIndex, Filters, FilterTypes } from '@models';

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
  if (fullIndexName == 'thorium_sample_tags') {
    return 'Tags';
  } else if (fullIndexName == 'thorium_sample_results') {
    return 'Results';
  } else {
    null;
  }
};

// map the selected index to the search indexes to use in our query
const mapSelectedIndex = (selectedIndex: string) => {
  switch (selectedIndex) {
    case 'All':
      return [ElasticIndex.SampleResults, ElasticIndex.SampleTags];
    case 'Tags':
      return [ElasticIndex.SampleTags];
    case 'Results':
      return [ElasticIndex.SampleResults];
    default:
      return new Array<ElasticIndex>();
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

const SearchResultsHeaders = () => {
  return (
    <Card className="basic-card panel">
      <Card.Body className="search-header">
        <Row>
          <Col className="d-flex justify-content-center sha256-col">SHA256</Col>
          <Col className="d-flex justify-content-center group-col hide-element">Group</Col>
          <Col className="d-flex justify-content-center index-col hide-element">Index</Col>
        </Row>
      </Card.Body>
    </Card>
  );
};

interface SearchResultItemProps {
  result: any;
  idx: number;
}

const SearchResultItem: React.FC<SearchResultItemProps> = ({ result, idx }) => {
  return (
    <Card className="panel">
      <Row>
        {/* add common relative spacing for sha, group, and index*/}
        <Col className="d-flex justify-content-center sha256-col">
          <Link className="hide-element-sha" to={`/file/${getSha256(result.id)}`}>
            {getSha256(result.id)}
          </Link>
          <Link className="hide-small-element-sha" to={`/file/${getSha256(result.id)}`}>
            {getSha256(result.id).substring(0, 10)}
          </Link>
        </Col>
        <Col className="d-flex justify-content-center group-col hide-element">{getGroup(result.id)}</Col>
        <Col className="d-flex justify-content-center index-col hide-element">{mapFullIndexName(result.index)}</Col>
        <hr />
      </Row>
      {result.highlight &&
        Object.keys(result.highlight).map(
          (key) =>
            key != 'group' && (
              <Row key={`${getSha256(result.id)}_${idx}_${key}`}>
                <Col>
                  <span>
                    {key}: {highlightResult(result.highlight[key])}
                  </span>
                </Col>
              </Row>
            ),
        )}
    </Card>
  );
};

// get repos using filters and an optional cursor
const getSearchResults = async (
  query: string,
  indexes: ElasticIndex[],
  filters: Filters,
  setSearchError: (error: string) => void,
  cursor: string | null,
): Promise<{ entitiesList: any[]; entitiesCursor: string | null }> => {
  if (query != '') {
    // get files list from API
    const { entityList, entityCursor } = await search(
      query.trim(),
      setSearchError,
      indexes,
      filters['groups'],
      filters['start'],
      filters['end'],
      cursor,
      filters['limit'],
    );
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

// component containing search bar and related functionality
export const Search = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [searching, setSearching] = useState<boolean>(false);
  const [filters, setFilters] = useState<SearchFilters>({ query: '' });
  const { userInfo } = useAuth();
  // the id of the cursor for paging search results;
  const [searchError, setSearchError] = useState<string>('');
  const [query, setQuery] = useState<string>('');
  const [debouncedQuery, setDebouncedQuery] = useState<string>('');
  const [selectedIndex, setSelectedIndex] = useState<string>('All');
  const [indexes, setIndexes] = useState<ElasticIndex[]>([]);

  // read filter values from url search query params
  const readURLSearchParams = () => {
    const savedIndexes: ElasticIndex[] = searchParams.getAll('indexes').map((index) => ElasticIndex[index as keyof typeof ElasticIndex]);
    // generate default selected groups list with each group set to unselected/false
    if (savedIndexes.length > 0) {
      setIndexes(savedIndexes);
    }
    const paramQuery = searchParams.get('query');
    //setDebouncedQuery(paramQuery ? paramQuery: "");
    setQuery(paramQuery ? paramQuery : '');
  };

  // update the selected index
  const updateSelectedIndex = (selectedIndex: string | null) => {
    if (!selectedIndex) {
      return;
    }
    // set string dropdown title
    setSelectedIndex(selectedIndex);
    // get indexes array from string key
    const indexes = mapSelectedIndex(selectedIndex);
    // set our indexes here and in our filters
    setIndexes(indexes);
    const pendingSearchFilters = structuredClone(filters);
    // when indexes have been set
    if (indexes && indexes.length > 0) {
      searchParams.set('indexes', indexes.toString());
      // set updated index in new search filters
      pendingSearchFilters['indexes'] = indexes;
      // when indexes are unset
    } else {
      // remove the indexes field from filters
      delete pendingSearchFilters['indexes'];
      searchParams.delete('indexes');
    }
    // now update searchFilters so page will update
    setFilters(pendingSearchFilters);
    // update url params in browser
    setSearchParams(searchParams);
  };

  const handleFilterUpdates = (newFilters: Filters) => {
    const pendingSearchFilters: SearchFilters = structuredClone(filters);
    if (newFilters.limit) {
      pendingSearchFilters.limit = newFilters.limit;
    }
    if (newFilters.groups) {
      pendingSearchFilters.groups = newFilters.groups;
    }
    if (newFilters.start) {
      pendingSearchFilters.start = newFilters.start;
    }
    if (newFilters.end) {
      pendingSearchFilters.end = newFilters.end;
    }
    setFilters(pendingSearchFilters);
  };

  // Update the search when any search parameter changes, after waiting for 0.5 seconds
  // this short delay allows the search to seem interactive/response without hammering the
  // API for every character change of the search query.
  useEffect(() => {
    const handleSetQuery = setTimeout(() => {
      setDebouncedQuery(query);
      // update query in url params
      setSearchError('');
      if (query != '') {
        searchParams.set('query', query);
      } else {
        searchParams.delete('query');
      }
      setSearchParams(searchParams);
      const pendingSearchFilters: SearchFilters = structuredClone(filters);
      pendingSearchFilters['query'] = query;
      setFilters(pendingSearchFilters);
    }, 500);
    return () => clearTimeout(handleSetQuery);
  }, [query]);

  useEffect(() => {
    readURLSearchParams();
  }, []);

  return (
    <Stack>
      <div className="d-flex flex-row justify-content-center">
        <Form className="d-flex search-bar justify-content-center position-relative">
          <Form.Control
            className="d-flex justify-content-center search-input"
            type="text"
            value={query}
            placeholder="Search data in Thorium"
            onChange={(e) => {
              setQuery(String(e.target.value));
              e.preventDefault();
            }}
            onKeyDown={(e) => {
              e.key === 'Enter' && e.preventDefault();
            }}
          />
          <IndexSelect index={selectedIndex} onChange={updateSelectedIndex} />
        </Form>
      </div>
      <BrowsingFilters
        title=""
        exclude={[FilterTypes.Tags]}
        onChange={handleFilterUpdates}
        groups={userInfo ? userInfo.groups : []}
        disabled={searching}
      />
      {query != '' && searchError == '' && (
        <Row>
          <EntityList
            type="Results"
            entityHeaders={<SearchResultsHeaders />}
            displayEntity={(result, idx) => <SearchResultItem result={result} idx={idx} />}
            filters={filters}
            fetchEntities={(filters, cursor) => getSearchResults(debouncedQuery, indexes, filters, setSearchError, cursor)}
            setLoading={setSearching}
            loading={searching}
          />
        </Row>
      )}
      {searchError && query != '' && (
        <Alert variant="danger" className="d-flex justify-content-center mt-1 mb-0">
          {searchError}
        </Alert>
      )}
    </Stack>
  );
};
