import { useEffect, useState } from 'react';
import { Button, Col, Row } from 'react-bootstrap';
import { useNavigate } from 'react-router';

// project imports
import { getCreatePathByEntity } from '@components/entities/create/EntityCreateRoutes';
import { OmnibarStandardTimeFilters } from '@components/pages/search/omnibar/Bars';
import { Clause, DefaultClausesEntities } from '@components/pages/search/omnibar/ClauseTypes';
import { TimeSelection } from '@components/pages/search/omnibar/timepicker/utils';
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
  const [clauses, setClauses] = useState<Clause[]>(DefaultClausesEntities());
  const [time, setTime] = useState<TimeSelection>({ mode: 'all' });

  // seed the parent's filters from the default clauses on mount so the initial list loads
  useEffect(() => {
    onChange(OmniClauseAndTimeToFilter(clauses, time));
  }, []);

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
          <OmnibarStandardTimeFilters
            clauses={clauses}
            setClauses={(next) => {
              setClauses(next);
              onChange(OmniClauseAndTimeToFilter(next, time));
            }}
            time={time}
            setTime={(next) => {
              setTime(next);
              onChange(OmniClauseAndTimeToFilter(clauses, next));
            }}
          />
        </Col>
      </Row>
    </>
  );
};

export default BrowsingFilters;
