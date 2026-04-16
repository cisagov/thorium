import React from 'react';
import { Link } from 'react-router-dom';
import { Row } from 'react-bootstrap';

// project imports
import { EntityBrowseConfig } from './config';
import {
  BrowsingCard,
  BrowsingContents,
  EntityGroups,
  EntityName,
  EntitySecondary,
  EntitySubmitters,
  LinkFields,
} from '@entities/browsing/shared';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import { listEntities } from '@thorpi/entities';
import { Filters } from '@models/search';
import { Entities } from '@models/entities/entities';
import { Other } from '@models/entities/other';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';

interface OtherItemProps {
  entity: Other;
}

const OtherItem: React.FC<OtherItemProps> = ({ entity }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`${getDetailsBasePathByEntity(Entities.Other)}/${entity.id}`} state={{ entity: entity }} className="no-decoration">
          <LinkFields>
            <EntityName>{entity.name}</EntityName>
            <EntitySecondary>{entity.created}</EntitySecondary>
            <EntityGroups>
              <small>
                <i>
                  {entity.submitter &&
                    (entity.groups.toString().length > 75
                      ? entity.groups.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                      : entity.groups.toString().replaceAll(',', ', '))}
                </i>
              </small>
            </EntityGroups>
            <EntitySubmitters>
              {entity.tags.submitter ? (
                <small>
                  <i>
                    {Object.keys(entity.tags.submitter).toString().length > 75
                      ? Object.keys(entity.tags.submitter).toString().replaceAll(',', ', ').substring(0, 75) + '...'
                      : Object.keys(entity.tags.submitter).toString().replaceAll(',', ', ')}
                  </i>
                </small>
              ) : null}
            </EntitySubmitters>
          </LinkFields>
        </Link>
        {entity.tags != undefined && <hr />}
        <Row>
          {entity.tags && Object.keys(entity.tags).length > 1 ? <CondensedEntityTags resource={Entities.Other} tags={entity.tags} /> : null}
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

const OtherListHeaders = () => (
  <BrowsingCard>
    <BrowsingContents>
      <Row>
        <EntityName>Name</EntityName>
        <EntitySecondary>Created</EntitySecondary>
        <EntityGroups>Group(s)</EntityGroups>
        <EntitySubmitters>Submitter(s)</EntitySubmitters>
      </Row>
    </BrowsingContents>
  </BrowsingCard>
);

const getOthers = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.Other];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return { entitiesList: entityList as Other[], entitiesCursor: entityCursor };
};

const OthersBrowsingConfig: EntityBrowseConfig<Entities.Other> = {
  docTitle: 'Others · Thorium',
  title: 'Others',
  typeLabel: '',
  kind: Entities.Other,
  creatable: true,
  entityHeaders: <OtherListHeaders />,
  renderEntity: (entity) => <OtherItem entity={entity} />,
  fetchEntities: getOthers,
};

export default OthersBrowsingConfig;
