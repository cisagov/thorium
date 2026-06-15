// spec: ../ToolResult.spec.md
import React, { useCallback, useState } from 'react';
import { FaCheck, FaChevronDown, FaChevronRight, FaPlus } from 'react-icons/fa6';
import { toast } from 'react-toastify';
import styled from 'styled-components';

// project imports
import { flattenEntityTags } from './entityTags';
import { ToolResultTabProps } from './types';
import EntityTypeIcon from '@components/entities/shared/EntityTypeIcon';
import AssociationKindModal from '@components/shared/AssociationKindModal';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { IconButton } from '@components/shared/buttons';
import { ButtonSize } from '@components/shared/buttons/types';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import EntitySummaryHover from '@components/shared/info/EntitySummaryHover';
import { entityRequestToInfo, SummaryPart } from '@components/shared/info/info';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import { buildEntityRequestForm } from '@components/entities/utilities';
import { createAssociation } from '@thorpi/associations';
import { createEntity } from '@thorpi/entities';
import { getResultEntities } from '@thorpi/results';
import { useAuth } from '@utilities/auth';
import { AssociationKind } from '@models/associations';
import { Entities, entityLabel } from '@models/entities';
import { EntityRequest } from '@models/entities/requests';

const KindList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
`;

const KindRow = styled.button`
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  text-align: left;
  padding: 6px 12px;
  background: var(--thorium-secondary-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 6px;
  color: var(--thorium-text);
  cursor: pointer;
  font-size: 0.95rem;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
  }
`;

const KindName = styled.span`
  font-weight: 600;
`;

const KindCount = styled.span`
  margin-left: auto;
  font-variant-numeric: tabular-nums;
  color: var(--thorium-secondary-text);
`;

const EntityList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 4px 0 4px 24px;
`;

const EntityRow = styled.div`
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 4px 12px;
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 6px;
`;

const EntityInfo = styled.div`
  min-width: 0;
  flex: 1 1 auto;
`;

const EntityName = styled.div`
  font-size: 0.9rem;
  color: var(--thorium-text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const CreatedTag = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--thorium-ok-bg);
  font-size: 0.8rem;
  flex: 0 0 auto;
`;

const TagPreview = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 2px;
`;

const TagChip = styled.span`
  font-size: 0.72rem;
  padding: 1px 6px;
  border-radius: 8px;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-secondary-text);
`;

/** Flatten an entity request's `key -> values[]` tags into `key: value` chips. */
function EntityTagPreview({ tags }: { tags: Record<string, string[]> }) {
  const chips = flattenEntityTags(tags);
  if (chips.length === 0) return null;
  return (
    <TagPreview>
      {chips.map((chip) => (
        <TagChip key={chip}>{chip}</TagChip>
      ))}
    </TagPreview>
  );
}

interface Prompt {
  kind: string;
  index: number;
}

/**
 * The "Entities" tab body: lists each entity kind (with count) from `result.entities`. Expanding a
 * kind lazily fetches the tool's entity structures for that kind and renders them, each with a
 * "Create" action that creates the entity and associates it with this file.
 */
const EntitiesTab: React.FC<ToolResultTabProps> = ({ result, sha256, tool }) => {
  const { checkCookie, userInfo } = useAuth();
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [entitiesByKind, setEntitiesByKind] = useState<Record<string, EntityRequest[]>>({});
  const [loadingKinds, setLoadingKinds] = useState<Set<string>>(new Set());
  const [created, setCreated] = useState<Set<string>>(new Set());
  const [creating, setCreating] = useState<Set<string>>(new Set());
  const [prompt, setPrompt] = useState<Prompt | null>(null);
  const [error, setError] = useState('');

  const errorHandler = useCallback(
    (e: string) => {
      setError(e);
      void checkCookie();
    },
    [checkCookie],
  );

  const toggleKind = useCallback(
    async (kind: string) => {
      setError('');
      const isOpen = expanded.has(kind);
      setExpanded((prev) => {
        const next = new Set(prev);
        if (isOpen) next.delete(kind);
        else next.add(kind);
        return next;
      });
      // lazily fetch the entities for this kind on first expand
      if (!isOpen && !entitiesByKind[kind]) {
        setLoadingKinds((prev) => new Set(prev).add(kind));
        const entities = await getResultEntities(sha256, tool, result.id, kind, errorHandler);
        setEntitiesByKind((prev) => ({ ...prev, [kind]: entities ?? [] }));
        setLoadingKinds((prev) => {
          const next = new Set(prev);
          next.delete(kind);
          return next;
        });
      }
    },
    [expanded, entitiesByKind, sha256, tool, result.id, errorHandler],
  );

  // create the prompted entity and associate it with this file using the chosen kind + groups
  const handleConfirm = useCallback(
    async (assocKind: AssociationKind, groups: string[]) => {
      if (!prompt) return;
      const req = entitiesByKind[prompt.kind]?.[prompt.index];
      const key = `${prompt.kind}:${prompt.index}`;
      setPrompt(null);
      if (!req) return;
      setError('');
      setCreating((prev) => new Set(prev).add(key));
      // use the groups chosen in the prompt for both the entity and the association
      const entity = await createEntity(buildEntityRequestForm({ ...req, groups }), errorHandler);
      if (entity?.id) {
        const assoc = await createAssociation(
          {
            kind: assocKind,
            source: { File: sha256 },
            targets: [{ Entity: { id: entity.id, name: req.name } }],
            groups,
            is_bidirectional: false,
          },
          errorHandler,
        );
        if (assoc) {
          setCreated((prev) => new Set(prev).add(key));
          toast(`Created "${req.name}" and associated it with this file.`);
        }
      }
      setCreating((prev) => {
        const next = new Set(prev);
        next.delete(key);
        return next;
      });
    },
    [prompt, entitiesByKind, sha256, errorHandler],
  );

  const kinds = Object.entries(result.entities ?? {});
  if (kinds.length === 0) {
    return <AlertBanner severity={Severity.Info}>This result has no entities.</AlertBanner>;
  }

  const promptName = prompt ? (entitiesByKind[prompt.kind]?.[prompt.index]?.name ?? 'entity') : '';
  // offer the user's member groups as options, always including the result's own groups
  const groupOptions = Array.from(new Set([...(result.groups ?? []), ...(userInfo?.groups ?? [])]));

  return (
    <>
      {error && <AlertBanner className="mb-2">{error}</AlertBanner>}
      <KindList>
        {kinds.map(([kind, count]) => {
          const isOpen = expanded.has(kind);
          const entities = entitiesByKind[kind] ?? [];
          return (
            <React.Fragment key={kind}>
              <KindRow type="button" onClick={() => void toggleKind(kind)} aria-expanded={isOpen}>
                {isOpen ? <FaChevronDown size={12} /> : <FaChevronRight size={12} />}
                <EntityTypeIcon kind={kind as Entities} />
                <KindName>{entityLabel(kind)}</KindName>
                <KindCount>{count}</KindCount>
              </KindRow>
              {isOpen && (
                <>
                  {loadingKinds.has(kind) && <LoadingSpinner loading={true} />}
                  {!loadingKinds.has(kind) && entities.length === 0 && (
                    <AlertBanner severity={Severity.Info}>No entities of this kind were found.</AlertBanner>
                  )}
                  {entities.length > 0 && (
                    <EntityList>
                      {entities.map((entity, index) => {
                        const key = `${kind}:${index}`;
                        const isCreated = created.has(key);
                        return (
                          <EntityRow key={key}>
                            <EntityInfo>
                              <EntitySummaryHover model={entityRequestToInfo(entity)} exclude={[SummaryPart.Title]}>
                                <EntityName title={entity.name}>{entity.name}</EntityName>
                              </EntitySummaryHover>
                              {entity.tags && <EntityTagPreview tags={entity.tags} />}
                            </EntityInfo>
                            {isCreated ? (
                              <CreatedTag>
                                <FaCheck size={12} /> Created
                              </CreatedTag>
                            ) : (
                              <OverlayTipTop tip="Create & associate">
                                <IconButton
                                  size={ButtonSize.Small}
                                  aria-label={`Create ${entity.name}`}
                                  disabled={creating.has(key)}
                                  onClick={() => setPrompt({ kind, index })}
                                >
                                  <FaPlus />
                                </IconButton>
                              </OverlayTipTop>
                            )}
                          </EntityRow>
                        );
                      })}
                    </EntityList>
                  )}
                </>
              )}
            </React.Fragment>
          );
        })}
      </KindList>
      <AssociationKindModal
        show={prompt !== null}
        entityName={promptName}
        defaultGroups={result.groups ?? []}
        groupOptions={groupOptions}
        onConfirm={(k, g) => void handleConfirm(k, g)}
        onCancel={() => setPrompt(null)}
      />
    </>
  );
};

export default EntitiesTab;
