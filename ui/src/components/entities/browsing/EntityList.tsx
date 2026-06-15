import React, { useEffect, useState, useRef } from 'react';
import { Col, Pagination, Row } from 'react-bootstrap';

// project imports
import { DEFAULT_LIST_LIMIT } from '../utilities';
import { LoadingSpinner } from '../../shared/fallback/LoadingSpinner';
import { Filters, SearchFilters } from '@models/search';
import AlertBanner from '@components/shared/alerts/AlertBanner';
import NoResultsBanner from '@components/shared/alerts/NoResultsBanner';

// spec: ./EntityBrowsing.spec.md

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

    const { entitiesList, entitiesCursor } = await fetchEntities(filters, requestCursor, setListError);

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
      // Skip the initial empty-filter render; fetch only once BrowsingFilters/omnibar
      // pushes real filters, so the list loads on first paint without a double-fetch.
      if (filters != null && Object.keys(filters).length > 0 && !loading) {
        void getEntityPage(true);
      }
    } else {
      isMountingRef.current = true;
    }
  }, [filters]);

  const updatePage = (page: number) => {
    if (page === maxPage && !loading) {
      void getEntityPage(false);
    }
    setPage(page);
  };

  const limit = filters.limit ? filters.limit : DEFAULT_LIST_LIMIT;

  const entityList = entities.slice(page * limit, page * limit + limit).map((entity, idx) => (
    <Row key={`${type}_entity_${idx}`} className="d-flex justify-content-center g-0">
      {displayEntity(entity, idx, filters)}
    </Row>
  ));

  return (
    <>
      <LoadingSpinner loading={loading} />
      {!loading && (
        <>
          <Row className="d-flex justify-content-center g-0">{entityHeaders}</Row>
          {entityList}
        </>
      )}
      {entities.length === 0 && !loading && isMountingRef.current && (
        <Row>
          <NoResultsBanner type={type} />
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
    </>
  );
};

export default EntityList;
