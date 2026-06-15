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
import { getDetailsBasePathByEntity } from '@components/entities/details/EntityDetailsRoutes';
import { formatAddress } from '@utilities/disassembly';
import { Entities, entityLabel } from '@models/entities/entities';
import { DecompiledFunction } from '@models/entities/functions';

// spec: ../EntityBrowsing.spec.md

const DecompiledFunctionListHeaders = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <EntityName>Name</EntityName>
          <EntitySecondary>Address · Tools</EntitySecondary>
          <EntityGroups>Group(s)</EntityGroups>
          <EntitySubmitters>Submitter(s)</EntitySubmitters>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface DecompiledFunctionItemProps {
  func: DecompiledFunction;
}

const DecompiledFunctionItem: React.FC<DecompiledFunctionItemProps> = ({ func }) => {
  const meta = func.metadata.DecompiledFunction;
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link
          to={`${getDetailsBasePathByEntity(Entities.DecompiledFunction)}/${func.id}`}
          state={{ decompiledFunction: func }}
          className="no-decoration"
        >
          <LinkFields>
            <EntityName>
              <EntityNameWithIcon entityId={func.id} hasImage={func.image != null}>
                {func.name}
              </EntityNameWithIcon>
            </EntityName>
            <EntitySecondary>{`${formatAddress(meta.address)}${meta.tools.length > 0 ? ` · ${meta.tools.join(', ')}` : ''}`}</EntitySecondary>
            <EntityGroups>
              <small>
                <i>{func.groups && func.groups.join(', ')}</i>
              </small>
            </EntityGroups>
            <EntitySubmitters>
              {func.submitter ? (
                <small>
                  <i>{func.submitter}</i>
                </small>
              ) : null}
            </EntitySubmitters>
          </LinkFields>
        </Link>
        {func.tags && Object.keys(func.tags).length > 1 && (
          <>
            <hr />
            <CondensedEntityTags resource={Entities.DecompiledFunction} tags={func.tags} />
          </>
        )}
      </BrowsingContents>
    </BrowsingCard>
  );
};

const DecompiledFunctionBrowsingConfig: EntityBrowseConfig<Entities.DecompiledFunction> = {
  docTitle: `${entityLabel(Entities.DecompiledFunction)}s · Thorium`,
  title: `${entityLabel(Entities.DecompiledFunction)}s`,
  typeLabel: entityLabel(Entities.DecompiledFunction),
  kind: Entities.DecompiledFunction,
  creatable: true,
  entityHeaders: <DecompiledFunctionListHeaders />,
  renderEntity: (entity) => <DecompiledFunctionItem func={entity} />,
  fetchEntities: makeEntityFetcher(Entities.DecompiledFunction),
};

export default DecompiledFunctionBrowsingConfig;
