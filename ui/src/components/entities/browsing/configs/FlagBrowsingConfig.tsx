import React from 'react';
import { Link } from 'react-router-dom';
import { Row } from 'react-bootstrap';

// project imports
import { EntityBrowseConfig, makeEntityFetcher } from './entityFetcher';
import {
  BrowsingCard,
  BrowsingContents,
  EntityGroups,
  EntityName,
  EntityNameWithIcon,
  EntitySecondary,
  EntitySubmitters,
  LinkFields,
} from '@entities/browsing/shared';
import CondensedEntityTags from '@components/tags/condensed/CondensedEntityTags';
import { Entities, entityLabel } from '@models/entities/entities';
import { Flag } from '@models/entities/flag';
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';

// spec: ../EntityBrowsing.spec.md

interface FlagItemProps {
  entity: Flag;
}

const FlagItem: React.FC<FlagItemProps> = ({ entity }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`${getDetailsBasePathByEntity(Entities.Flag)}/${entity.id}`} state={{ entity: entity }} className="no-decoration">
          <LinkFields>
            <EntityName>
              <EntityNameWithIcon entityId={entity.id} hasImage={entity.image != null}>
                {entity.name}
              </EntityNameWithIcon>
            </EntityName>
            <EntitySecondary>{entity.metadata.Flag?.confidence}</EntitySecondary>
            <EntityGroups>
              <small>
                <i>{entity.groups.toString().replaceAll(',', ', ')}</i>
              </small>
            </EntityGroups>
            <EntitySubmitters>
              <small>
                <i>{entity.submitter}</i>
              </small>
            </EntitySubmitters>
          </LinkFields>
        </Link>
        {entity.tags && Object.keys(entity.tags).length > 1 && (
          <>
            <hr />
            <CondensedEntityTags resource={Entities.Flag} tags={entity.tags} />
          </>
        )}
      </BrowsingContents>
    </BrowsingCard>
  );
};

const FlagListHeaders = () => (
  <BrowsingCard>
    <BrowsingContents>
      <Row>
        <EntityName>Name</EntityName>
        <EntitySecondary>Confidence</EntitySecondary>
        <EntityGroups>Group(s)</EntityGroups>
        <EntitySubmitters>Submitter(s)</EntitySubmitters>
      </Row>
    </BrowsingContents>
  </BrowsingCard>
);

const FlagBrowsingConfig: EntityBrowseConfig<Entities.Flag> = {
  docTitle: 'Flags · Thorium',
  title: 'Flags',
  typeLabel: entityLabel(Entities.Flag),
  kind: Entities.Flag,
  creatable: true,
  entityHeaders: <FlagListHeaders />,
  renderEntity: (entity) => <FlagItem entity={entity} />,
  fetchEntities: makeEntityFetcher(Entities.Flag),
};

export default FlagBrowsingConfig;
