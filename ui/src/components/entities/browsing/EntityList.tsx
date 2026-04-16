import React, { useEffect, useState, Fragment, useRef } from 'react';
import { Col, Pagination, Row } from 'react-bootstrap';

// project imports
import { DEFAULT_LIST_LIMIT } from '../utilities';
import { LoadingSpinner } from '../../shared/fallback/LoadingSpinner';
import { Filters, SearchFilters } from '@models/search';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';

interface EntityListProps<T> {
  type: string;
  displayEntity: (entity: T, idx: number, filters?: Filters) => React.ReactNode;
  entityHeaders: React.ReactNode;
  filters: SearchFilters | Filters;
  fetchEntities: (
    filters: Filters,
    cursor: string | null,
    errorHandler: (error: string) => void,
  ) => Promise<{ entitiesList: T[]; entitiesCursor: string | null }>;
  loading: boolean;
  setLoading: (loading: boolean) => void;
}

const EntityList = <T,>({ type, displayEntity, entityHeaders, filters, fetchEntities, loading, setLoading }: EntityListProps<T>) => {
  const [entities, setEntities] = useState<T[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [listError, setListError] = useState('');
  const [page, setPage] = useState(0);
  const [maxPage, setMaxPage] = useState(1);

  const getEntityPage = async (reset: boolean) => {
    setLoading(true);

    let requestCursor = cursor;
    if (reset) {
      setPage(0);
      requestCursor = null;
    }

    setListError('');

    const { entitiesList, entitiesCursor } = await fetchEntities(filters as Filters, requestCursor, setListError);

    setCursor(entitiesCursor);
    setLoading(false);

    let allEntities: T[] = [];
    if (reset) {
      allEntities = entitiesList;
    } else {
      allEntities = [...entities, ...entitiesList];
    }

    const limit = filters.limit ? filters.limit : DEFAULT_LIST_LIMIT;
    setMaxPage(Math.ceil(allEntities.length / limit));
    setEntities(allEntities);
  };

  const isMountingRef = useRef(false);

  useEffect(() => {
    isMountingRef.current = true;
  }, []);

  useEffect(() => {
    if (isMountingRef.current) {
      if (filters != null && Object.keys(filters).length > 0 && !loading) {
        getEntityPage(true);
      }
    } else {
      isMountingRef.current = true;
    }
  }, [filters]);

  const updatePage = (page: number) => {
    if (page === maxPage && !loading) {
      getEntityPage(false);
    }
    setPage(page);
  };

  const limit = filters.limit ? filters.limit : DEFAULT_LIST_LIMIT;

  return (
    <Fragment>
      {!loading && <Row className="d-flex justify-content-center">{entityHeaders}</Row>}

      {!loading &&
        entities.slice(page * limit, page * limit + limit).map((entity, idx) => (
          <Row key={`${type}_entity_${idx}`} className="d-flex justify-content-center">
            {displayEntity(entity, idx, filters as Filters)}
          </Row>
        ))}
      <LoadingSpinner loading={loading} />
      {entities.length === 0 && !loading && isMountingRef.current && (
        <Row>
          <AlertBanner severity={Severity.Info} className="m-1">
            {type ? <>No {type} Found</> : <>None Found</>}
          </AlertBanner>
        </Row>
      )}
      {listError != '' && <AlertBanner className="m-1">{listError}</AlertBanner>}
      {entities.length > 0 && (
        <Row className="mt-3">
          <Col className="d-flex justify-content-center">
            <Pagination>
              <Pagination.Item onClick={() => updatePage(page - 1)} disabled={page === 0}>
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

export default EntityList;
