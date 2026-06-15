// spec: ./EntityBrowser.spec.md
import React, { useRef, useState } from 'react';
import { FaAngleDown, FaAngleUp } from 'react-icons/fa6';

// project imports
import { MetadataContent, MetadataSection, MetadataToggleRow } from './EntityBrowser.styled';
import { ExpandToggle } from '@components/shared/buttons/ExpandToggle';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import EntitySummary, { SummaryVariant } from '@components/shared/info/EntitySummary';
import { InfoModel, SummaryPart, treeNodeToInfo } from '@components/shared/info/info';
import { getEntity } from '@thorpi/entities';
import { TreeNodeKey } from '@models/trees';

interface MetadataBoxProps {
  /** The info model built from the graph/tree node — rendered immediately (and as the fetch fallback). */
  model: InfoModel;
  /**
   * When set, this row is an **entity** node with the given id. The graph/tree node omits an entity's heavy
   * content (a SigmaRule's `rule` YAML, a CompiledFunction's `disassembly`, a DecompiledFunction's decompiled
   * `content`), so on first expand the full entity is fetched and rendered in its place. Omitted for
   * File/Repo/Tag nodes, which already carry everything.
   */
  entityId?: string;
  /** Whether the details body is expanded. Controlled by the parent ({@link EntityRow}) so it can suppress the
   * header's hover preview while the details are open. */
  expanded: boolean;
  /** Called with the requested next expanded state when the "details" caret is toggled. */
  onExpandedChange: (next: boolean) => void;
}

/**
 * Condensed metadata affordance under a row's header: a single "details" up/down caret (no preview peek),
 * collapsed by default with minimal vertical footprint. Expanding reveals the node's full metadata via the
 * shared {@link EntitySummary} (kind/title omitted — the header already shows the name; the duplicate marker
 * lives on the header, so it's suppressed here too). Its expanded state is owned by the parent (controlled), so
 * the row can hide its hover summary preview while these details are open.
 *
 * For **entity** nodes ({@link MetadataBoxProps.entityId} set), the graph/tree node carries only lightweight
 * metadata, so the body would be missing the entity's rich content. On the first expand we lazily
 * `getEntity(id)` (once, cached for the row's lifetime) and rebuild the model from that authoritative record,
 * so sigma-rule YAML / disassembly / decompiled source and every other field render in the body. The fetch
 * falls back to the graph-node `model` on failure; a spinner shows while it's in flight.
 */
const MetadataBox: React.FC<MetadataBoxProps> = ({ model, entityId, expanded, onExpandedChange }) => {
  // the richer model built from the lazily-fetched full entity (null until fetched); rendered in place of the
  // graph-node `model` once available
  const [fullModel, setFullModel] = useState<InfoModel | null>(null);
  const [loading, setLoading] = useState(false);
  // fetch the full entity at most once per row (survives collapse/re-expand while the row stays mounted)
  const fetchedRef = useRef(false);

  const onToggle = () => {
    const next = !expanded;
    onExpandedChange(next);
    if (next && entityId && !fetchedRef.current) {
      fetchedRef.current = true;
      setLoading(true);
      // pull the authoritative entity so its heavy content (rule/disassembly/decompilation) shows; on failure
      // we simply keep the graph-node model (no error surface needed in this condensed affordance)
      void getEntity(entityId, () => {})
        .then((entity) => {
          if (entity) setFullModel(treeNodeToInfo({ [TreeNodeKey.Entity]: entity }));
        })
        .finally(() => setLoading(false));
    }
  };

  return (
    <MetadataSection>
      <MetadataToggleRow>
        <ExpandToggle data-testid="entity-details-toggle" aria-expanded={expanded} onClick={onToggle}>
          {expanded ? <FaAngleUp /> : <FaAngleDown />} details
        </ExpandToggle>
      </MetadataToggleRow>
      {expanded && (
        <MetadataContent>
          <EntitySummary model={fullModel ?? model} variant={SummaryVariant.Compact} exclude={[SummaryPart.Kind, SummaryPart.Title]} />
          {loading && <LoadingSpinner loading={true} />}
        </MetadataContent>
      )}
    </MetadataSection>
  );
};

export default MetadataBox;
