import React, { useEffect, useState, Fragment, useRef } from 'react';
import { Alert, Col, Pagination, Row } from 'react-bootstrap';

// project imports
import { LoadingSpinner, DEFAULT_LIST_LIMIT } from '@components';
import { Filters, SearchFilters } from '@models';

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
    if (!isMountingRef.current) {
      if (filters != null) {
        getEntityPage(true);
      }
    } else {
      isMountingRef.current = false;
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
      <Row className="d-flex justify-content-center">{entityHeaders}</Row>
      {!loading &&
        entities.slice(page * limit, page * limit + limit).map((entity, idx) => (
          <Row key={`${type}_entity_${idx}`} className="d-flex justify-content-center">
            {displayEntity(entity, idx)}
          </Row>
        ))}
      <LoadingSpinner loading={loading} />
      {entities.length == 0 && !loading && (
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
