import { useRef, useState } from 'react';
import { Button, Col, Row } from 'react-bootstrap';
import { useNavigate } from 'react-router';
import { FaFilter } from 'react-icons/fa';

// project imports
import FilterFields from './FilterFields';
import { Placement, PositionType } from '@components/shared/windows';
import { OverlayTipLeft, OverlayTipRight } from '@components/shared/overlay/tips';
import { OverlayWindow } from '@components/shared/windows/OverlayWindow';
import Title from '@components/shared/titles/Title';
import { Entities } from '@models/entities';
import { Filters, FilterTypes } from '@models/search';
import { getCreatePathByEntity } from '@components/entities/create/EntityCreateRoutes';

interface BrowsingFiltersProps {
  onChange: (filters: Filters) => void; // call back to change filters
  disabled?: boolean; // whether changes to filters are disabled
  title?: string; // name of entity type being listed
  groups: Array<string>; // the groups a user can select from
  exclude?: FilterTypes[];
  creatable?: boolean; // link to create page with button
  kind?: Entities;
}

const BrowsingFilters: React.FC<BrowsingFiltersProps> = ({
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
      <Row className="align-items-center">
        <Col />
        <Col className="text-center">
          <div className="d-inline-flex align-items-center justify-content-center gap-2">
            {title && <Title className="m-0">{title}</Title>}
            <OverlayTipRight tip={`${hideFilters ? 'Expand' : 'Hide'} filters`}>
              <Button ref={filterRef} variant="" className="clear-btn p-0 border-0" onClick={() => setHideFilters(!hideFilters)}>
                <FaFilter size="18" />
              </Button>
            </OverlayTipRight>
          </div>
        </Col>
        <Col className="d-flex justify-content-end">
          {creatable && (
            <OverlayTipLeft tip={`Create a new ${kind}.`}>
              <Button
                className="ok-btn my-3"
                variant=""
                disabled={disabled}
                onClick={() => navigate(`${getCreatePathByEntity(kind ? kind : Entities.Other)}`)}
              >
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

export default BrowsingFilters;
