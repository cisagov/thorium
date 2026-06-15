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
import { CompiledFunction } from '@models/entities/functions';

// spec: ../EntityBrowsing.spec.md

const CompiledFunctionListHeaders = () => {
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Row>
          <EntityName>Name</EntityName>
          <EntitySecondary>Address · Instructions</EntitySecondary>
          <EntityGroups>Group(s)</EntityGroups>
          <EntitySubmitters>Submitter(s)</EntitySubmitters>
        </Row>
      </BrowsingContents>
    </BrowsingCard>
  );
};

interface CompiledFunctionItemProps {
  func: CompiledFunction;
}

const CompiledFunctionItem: React.FC<CompiledFunctionItemProps> = ({ func }) => {
  const meta = func.metadata.CompiledFunction;
  return (
    <BrowsingCard>
      <BrowsingContents>
        <Link
          to={`${getDetailsBasePathByEntity(Entities.CompiledFunction)}/${func.id}`}
          state={{ compiledFunction: func }}
          className="no-decoration"
        >
          <LinkFields>
            <EntityName>
              <EntityNameWithIcon entityId={func.id} hasImage={func.image != null}>
                {func.name}
              </EntityNameWithIcon>
            </EntityName>
            <EntitySecondary>{`${formatAddress(meta.address)} · ${meta.disassembly.length} instr`}</EntitySecondary>
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
            <CondensedEntityTags resource={Entities.CompiledFunction} tags={func.tags} />
          </>
        )}
      </BrowsingContents>
    </BrowsingCard>
  );
};

const CompiledFunctionBrowsingConfig: EntityBrowseConfig<Entities.CompiledFunction> = {
  docTitle: `${entityLabel(Entities.CompiledFunction)}s · Thorium`,
  title: `${entityLabel(Entities.CompiledFunction)}s`,
  typeLabel: entityLabel(Entities.CompiledFunction),
  kind: Entities.CompiledFunction,
  creatable: true,
  entityHeaders: <CompiledFunctionListHeaders />,
  renderEntity: (entity) => <CompiledFunctionItem func={entity} />,
  fetchEntities: makeEntityFetcher(Entities.CompiledFunction),
};

export default CompiledFunctionBrowsingConfig;
