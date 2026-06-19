import { useEffect } from 'react';
import { Button, Col, Row } from 'react-bootstrap';
import { useNavigate } from 'react-router';

// project imports
import { getCreatePathByEntity } from '@components/entities/create/EntityCreateRoutes';
import { OmnibarStandardTimeFilters } from '@components/shared/inputs/omnibar/Bars';
import { DefaultClausesEntities } from '@components/shared/inputs/omnibar/ClauseTypes';
import { useOmnibarUrlState } from '@components/shared/inputs/omnibar/useOmnibarUrlState';
import { OverlayTipLeft } from '@components/shared/overlay/tips';
import Title from '@components/shared/titles/Title';
import { OmniClauseAndTimeToFilter } from '@utilities/search';
import { Entities } from '@models/entities';
import { Filters } from '@models/search';

interface BrowsingFiltersProps {
  onChange: (filters: Filters) => void; // call back to change filters
  disabled?: boolean; // whether changes to filters are disabled
  title?: string; // name of entity type being listed
  creatable?: boolean; // link to create page with button
  kind?: Entities;
}

const BrowsingFilters: React.FC<BrowsingFiltersProps> = ({ onChange, disabled = false, title = null, kind, creatable = false }) => {
  const navigate = useNavigate();
  // clauses + time live in the URL so filtered browse views are shareable/bookmarkable
  const { clauses, setClauses, time, setTime } = useOmnibarUrlState({ clauses: DefaultClausesEntities(), time: { mode: 'all' } });

  // push filters to the parent on mount and whenever the URL-backed clauses/time change (covers
  // user edits, deep links, and back/forward navigation uniformly)
  useEffect(() => {
    onChange(OmniClauseAndTimeToFilter(clauses, time));
  }, [clauses, time, onChange]);

  return (
    <>
      <Row className="align-items-center">
        <Col />
        <Col className="text-center">
          <div className="d-inline-flex align-items-center justify-content-center gap-2">
            {title && <Title className="m-0">{title}</Title>}
          </div>
        </Col>
        <Col className="d-flex justify-content-end">
          {creatable && (
            <OverlayTipLeft tip={`Create a new ${kind}.`}>
              <Button
                className="ok-btn my-3"
                variant=""
                disabled={disabled}
                onClick={() => void navigate(`${getCreatePathByEntity(kind ? kind : Entities.Other)}`)}
              >
                <b>+</b>
              </Button>
            </OverlayTipLeft>
          )}
        </Col>
      </Row>
      <Row>
        <Col className="d-flex justify-content-center">
          <OmnibarStandardTimeFilters clauses={clauses} setClauses={setClauses} time={time} setTime={setTime} />
        </Col>
      </Row>
    </>
  );
};

export default BrowsingFilters;
