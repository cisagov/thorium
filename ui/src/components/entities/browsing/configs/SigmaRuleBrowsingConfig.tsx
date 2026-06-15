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
import FieldBadge from '@components/shared/badges/FieldBadge';
import { listEntities } from '@thorpi/entities';
import { Filters } from '@models/search';
import { Entities } from '@models/entities/entities';
import { SigmaRule } from '@models/entities/rules/sigma';
import { getDetailsBasePathByEntity } from '../../details/EntityDetailsRoutes';

// spec: ../EntityBrowsing.spec.md

interface SigmaRuleItemProps {
  sigma: SigmaRule;
}

const SigmaRuleItem: React.FC<SigmaRuleItemProps> = ({ sigma }) => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link to={`${getDetailsBasePathByEntity(Entities.SigmaRule)}/${sigma.id}`} state={{ sigma: sigma }} className="no-decoration">
          <LinkFields>
            <EntityName>{sigma.name}</EntityName>
            <EntitySecondary>
              {sigma.metadata.SigmaRule.applies_to.map((target, idx) => (
                <FieldBadge key={`${target}_${idx}`} color="gray" field={`${target}`} />
              ))}
            </EntitySecondary>
            <EntityGroups>
              <small>
                <i>
                  {sigma.groups &&
                    (sigma.groups.toString().length > 75
                      ? sigma.groups.toString().replaceAll(',', ', ').substring(0, 75) + '...'
                      : sigma.groups.toString().replaceAll(',', ', '))}
                </i>
              </small>
            </EntityGroups>
            <EntitySubmitters>{sigma.submitter}</EntitySubmitters>
          </LinkFields>
        </Link>
        {sigma.tags != undefined && <hr />}
        {sigma.tags && Object.keys(sigma.tags).length > 1 ? <CondensedEntityTags resource={Entities.SigmaRule} tags={sigma.tags} /> : null}
      </BrowsingContents>
    </BrowsingCard>
  );
};

const SigmaRuleListHeaders = () => (
  <BrowsingCard>
    <BrowsingContents>
      <Row>
        <EntityName>Name</EntityName>
        <EntitySecondary>Targets</EntitySecondary>
        <EntityGroups>Groups(s)</EntityGroups>
        <EntitySubmitters>Submitter</EntitySubmitters>
      </Row>
    </BrowsingContents>
  </BrowsingCard>
);

const getSigmaRules = async (filters: Filters, cursor: string | null, errorHandler: (error: string) => void) => {
  const listFilters = structuredClone(filters);
  listFilters.kinds = [Entities.SigmaRule];
  const { entityList, entityCursor } = await listEntities(listFilters, errorHandler, true, cursor);
  return { entitiesList: entityList as SigmaRule[], entitiesCursor: entityCursor };
};

const SigmaRulesBrowsingConfig: EntityBrowseConfig<Entities.SigmaRule> = {
  docTitle: 'Sigma Rules · Thorium',
  title: 'Sigma Rules',
  typeLabel: '',
  kind: Entities.SigmaRule,
  creatable: true,
  entityHeaders: <SigmaRuleListHeaders />,
  renderEntity: (entity) => <SigmaRuleItem sigma={entity} />,
  fetchEntities: getSigmaRules,
};

export default SigmaRulesBrowsingConfig;
