import React, { useEffect, useState, Fragment, useRef } from 'react';
import { Alert, Card, Col, Pagination, Row } from 'react-bootstrap';
import styled from 'styled-components';

// project imports
import { scaling } from '@styles';
import { LoadingSpinner, DEFAULT_LIST_LIMIT } from '@components';
import { Filters, SearchFilters } from '@models';

export const BrowsingCard = styled(Card)`
  margin: 0.1em 0em 0em 0.1em;
  min-height: 3em;
  color: var(--thorium-text);
  background-color: var(--thorium-panel-bg);
`;

export const BrowsingContents = styled(Card.Body)`
  flex-wrap: wrap;
  flex: 1 1 auto;
  padding: var(--bs-card-spacer-y) var(--bs-card-spacer-x);
  color: var(--bs-card-color);
`;

export const LinkFields = styled(Row)`
  display: flex;
  flex-wrap: wrap;
  color: var(--thorium-text);
  background-color: var(--thorium-panel-bg);

  &:hover {
    color: var(--thorium-text);
    background-color: var(--thorium-highlight-panel-bg);
    box-shadow:
      inset 0 0 1px var(--thorium-panel-border),
      0 0 4px var(--thorium-highlight-panel-border) !important;
    // this makes sure the box shadow isn't hidden behind the card above
    z-index: 1000;
  }
`;

export const EntityName = styled(Col)`
  white-space: pre-wrap;
  text-align: center;
  flex-wrap: wrap;
  word-break: break-all;
  min-width: 400px;
  color: var(--thorium-text);
`;

export const EntityOrigin = styled(Col)`
  min-width: 300px;
  text-align: center;
  color: var(--thorium-text);
  @media (max-width: ${scaling.lg}) {
    display: none !important;
  }
`;

export const EntitySubmitters = styled(Col)`
  flex-wrap: wrap;
  text-align: center;
  min-width: 150px;
  color: var(--thorium-text);
  @media (max-width: ${scaling.xxl}) {
    display: none !important;
  }
`;

export const EntityGroups = styled(Col)`
  flex-wrap: wrap;
  text-align: center;
  min-width: 200px;
  color: var(--thorium-text);
  @media (max-width: ${scaling.xl}) {
    display: none !important;
  }
`;

interface EntityListProps {
  type: string; // type of entity, used as a title or in alerts
  displayEntity: (entity: any, idx: number) => React.JSX.Element; // display call
  entityHeaders: React.ReactElement;
  filters: SearchFilters | Filters;
  fetchEntities: (
    filters: Filters,
    cursor: string | null,
    errorHandler: (error: string) => void,
  ) => Promise<{ entitiesList: any[]; entitiesCursor: string | null }>;
  loading: boolean;
  setLoading: (loading: boolean) => void;
}

export const EntityList: React.FC<EntityListProps> = ({
  type,
  displayEntity,
  entityHeaders,
  filters,
  fetchEntities,
  loading,
  setLoading,
}) => {
  const [entities, setEntities] = useState<any[]>([]);
  // paging/cursor values
  const [cursor, setCursor] = useState<string | null>(null);
  const [listError, setListError] = useState('');
  const [page, setPage] = useState(0);
  const [maxPage, setMaxPage] = useState(1);

  // Get an entity list using set filters
  const getEntityPage = async (reset: boolean) => {
    setLoading(true);
    let requestCursor = cursor;
    if (reset) {
      setPage(0);
      requestCursor = null;
    }
    setListError('');
    // get more entity items and updated cursor
    const { entitiesList, entitiesCursor } = await fetchEntities(filters, requestCursor, setListError);
    setCursor(entitiesCursor);
    // API responded, no longer waiting
    setLoading(false);
    // save any listed entities if request was successful
    let allEntities = [];
    if (reset) {
      allEntities = entitiesList;
    } else {
      allEntities = [...entities, ...entitiesList];
    }
    const limit = filters.limit ? filters.limit : DEFAULT_LIST_LIMIT;
    setMaxPage(Math.ceil(allEntities.length / limit));
    setEntities(allEntities);
  };

  // don't render on first mount, wait for url params to be read in and set
  const isMountingRef = useRef(false);

  useEffect(() => {
    isMountingRef.current = true;
  }, []);

  // get new entity list whenever filters changes
  useEffect(() => {
    if (isMountingRef.current) {
      // on initial page load filters will be {}, don't load as this
      // prevents load once filters are read from URL params
      if (filters != null && Object.keys(filters).length > 0 && !loading) {
        getEntityPage(true);
      }
    } else {
      isMountingRef.current = true;
    }
  }, [filters]);

  // update the displayed page and retrieve new results when end of
  // already fetched results is reached
  const updatePage = (page: number) => {
    // Trigger getting more entities when paging past end of already
    // retrieved entities
    if (page == maxPage && !loading) {
      getEntityPage(false);
    }
    setPage(page);
  };

  // limit may not be set in the filter initially
  const limit = filters.limit ? filters.limit : DEFAULT_LIST_LIMIT;
  return (
    <Fragment>
      {!loading && <Row className="d-flex justify-content-center">{entityHeaders}</Row>}
      {!loading &&
        entities.slice(page * limit, page * limit + limit).map((entity, idx) => (
          <Row key={`${type}_entity_${idx}`} className="d-flex justify-content-center">
            {displayEntity(entity, idx)}
          </Row>
        ))}
      <LoadingSpinner loading={loading} />
      {entities.length == 0 && !loading && isMountingRef.current && (
        <Row>
          <Alert variant="info" className="d-flex justify-content-center m-1">
            {type ? <>No {type} Found</> : <>None Found</>}
          </Alert>
        </Row>
      )}
      {listError != '' && (
        <Alert variant="danger" className="d-flex justify-content-center m-1">
          {listError}
        </Alert>
      )}
      {entities.length > 0 && (
        <Row className="mt-3">
          <Col className="d-flex justify-content-center">
            <Pagination>
              <Pagination.Item onClick={() => updatePage(page - 1)} disabled={page == 0}>
                Back
              </Pagination.Item>
              <Pagination.Item onClick={() => updatePage(page + 1)} disabled={!cursor && page + 1 >= maxPage}>
                Next
              </Pagination.Item>
            </Pagination>
          </Col>
        </Row>
      )}
    </Fragment>
  );
};
