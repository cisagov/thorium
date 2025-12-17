import { useRef, useState } from 'react';
import { Button, Col, Row } from 'react-bootstrap';
import { useNavigate } from 'react-router';
import { FaFilter } from 'react-icons/fa';

// project imports
import { OverlayTipLeft, OverlayTipRight, OverlayWindow, Placement, PositionType, Title } from '@components';
import { Entities, Filters, FilterTypes } from '@models';
import { FilterFields } from './filter_fields';

interface BrowsingFiltersProps {
  onChange: (filters: Filters) => void; // call back to change filters
  disabled?: boolean; // whether changes to filters are disabled
  title?: string; // name of entity type being listed
  groups: Array<string>; // the groups a user can select from
  exclude?: FilterTypes[];
  creatable?: boolean; // link to create page with button
  kind?: Entities;
}

export const BrowsingFilters: React.FC<BrowsingFiltersProps> = ({
  onChange,
  groups,
  disabled = false,
  title = null,
  exclude = [],
  kind,
  creatable = false,
}) => {
  const navigate = useNavigate();
  // show filters or don't
  const [hideFilters, setHideFilters] = useState(true);

  // create ref for positioning filter window
  const filterRef = useRef(null);
  return (
    <>
      <Row>
        <Col />
        <Col className="d-flex justify-content-center">
          {title && <Title>{title}</Title>}
          <OverlayTipRight tip={`${hideFilters ? 'Expand' : 'Hide'} filters`}>
            <Button ref={filterRef} variant="" className="mt-3 clear-btn" onClick={() => setHideFilters(!hideFilters)}>
              <FaFilter size="18" />
            </Button>
          </OverlayTipRight>
        </Col>
        <Col className="d-flex justify-content-end">
          {creatable && (
            <OverlayTipLeft tip={`Create a new ${kind}.`}>
              <Button className="ok-btn my-3" variant="" disabled={disabled} onClick={() => navigate(`/create/${kind?.toLowerCase()}`)}>
                <b>+</b>
              </Button>
            </OverlayTipLeft>
          )}
        </Col>
      </Row>
      <OverlayWindow
        title="Filters"
        placement={Placement.BottomRight}
        show={!hideFilters}
        onHide={() => setHideFilters(true)}
        width={500}
        height={460}
        positioning={PositionType.Fixed}
        parentRef={filterRef}
      >
        <FilterFields show={!hideFilters} disabled={disabled} onChange={onChange} exclude={exclude} groups={groups} />
      </OverlayWindow>
    </>
  );
};
