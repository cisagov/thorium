import React from 'react';
import { Link } from 'react-router-dom';
import styled from 'styled-components';

// project imports
import Collapsible from './Collapsible';
import CollapsibleMarkdown from './CollapsibleMarkdown';
import { applyExclusions, FieldLayout, FieldRender, InfoField, InfoModel, InfoNote, InfoNoteLevel, InfoSection } from './info';
import FieldBadge from '@components/shared/badges/FieldBadge';
import Time from '@components/shared/Time';
import Subtitle from '@components/shared/titles/Subtitle';
import { bucketTags, filterExcludedTags, filterIncludedTags } from '@components/tags/utilities';
import TagBadge from '@components/tags/TagBadge';
import { Entities } from '@models/entities';
import { Tags } from '@models/tags';

// spec: ./SPEC.md

/** How compactly a summary renders. Chosen per surface (hover = compact, side panel = expanded). */
export enum SummaryVariant {
  /** Dense, no delimiters, small font — hover popovers. */
  Compact = 'compact',
  /** Whitespace + section headings + `<hr>` delimiters — the graph side panel. */
  Expanded = 'expanded',
}

interface FilteredNodeTagsProps {
  tags: Tags;
  /** When set, only these tag keys are rendered. */
  includeKeys?: string[];
  /** When set, these tag keys are removed before rendering. */
  excludeKeys?: string[];
  /** When set, each tag links to this resource's browse page filtered by the tag; else non-clickable. */
  resource?: Entities;
}

/**
 * Render a node/entity's tags grouped by significance (danger, ATT&CK, MBC, file-info, general).
 *
 * `includeKeys`/`excludeKeys` narrow which tag keys render so a section can scope its own tag subset
 * (with the catch-all block excluding those keys). With neither set, the default grouping renders all
 * tags except the always-excluded provenance keys. When `resource` is set the tags become browse links.
 */
export const FilteredNodeTags: React.FC<FilteredNodeTagsProps> = ({ tags, includeKeys, excludeKeys, resource }) => {
  let scoped = tags;
  if (includeKeys) scoped = filterIncludedTags(scoped, includeKeys);
  if (excludeKeys) scoped = filterExcludedTags(scoped, excludeKeys);

  const { danger, attack, mbc, fileInfo, general } = bucketTags(scoped);
  const allTags = [danger, attack, mbc, fileInfo, general];
  const action = resource ? 'link' : 'none';

  return (
    <>
      {allTags.map((tagGrouping) =>
        Object.keys(tagGrouping)
          .sort()
          .map((tagKey) =>
            Object.keys(tagGrouping[tagKey])
              .sort()
              .map((tagValue) => (
                <TagBadge
                  key={`${tagKey}_${tagValue}`}
                  tag={tagKey}
                  value={tagValue}
                  condensed={true}
                  action={action}
                  resource={resource}
                />
              )),
          ),
      )}
    </>
  );
};

const Wrapper = styled.div<{ $variant: SummaryVariant }>`
  margin: ${({ $variant }) => ($variant === SummaryVariant.Compact ? '0' : '0.5rem')};
  font-size: ${({ $variant }) => ($variant === SummaryVariant.Compact ? '0.82rem' : 'inherit')};
  max-width: 100%;
  word-break: break-word;
`;

const KindLabel = styled(Subtitle)`
  text-transform: uppercase;
`;

const TitleLink = styled(Link)`
  color: var(--thorium-link-text);
  word-break: break-all;
  &:hover {
    color: var(--thorium-highlight-text);
  }
`;

const TitleText = styled.span`
  color: var(--thorium-text);
  word-break: break-all;
`;

// a named-entity reference rendered as a chip that links to that entity's details page
const LinkChip = styled(Link)`
  display: inline-block;
  margin: 0 4px 4px 0;
  padding: 0.2em 0.5em;
  border-radius: 6px;
  font-size: 0.85em;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-link-text);
  text-decoration: none;
  word-break: break-word;

  &:hover {
    color: var(--thorium-highlight-text);
    background: var(--thorium-highlight-panel-bg);
  }
`;

// same chip shape for a name we can't link (no known details path)
const PlainChip = styled.span`
  display: inline-block;
  margin: 0 4px 4px 0;
  padding: 0.2em 0.5em;
  border-radius: 6px;
  font-size: 0.85em;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  word-break: break-word;
`;

const Description = styled.div`
  margin-top: 4px;
  color: var(--thorium-secondary-text);
`;

// a full path / long name: italicized inline text that breaks at any character so it flows after the
// label and wraps back to the section's left edge (inline, not a block column)
const PathValue = styled.span`
  font-style: italic;
  word-break: break-all;
  overflow-wrap: anywhere;
`;

// generic inline text value (raw numbers, joined lists) that breaks by character
const TextValue = styled.span`
  word-break: break-all;
  overflow-wrap: anywhere;
`;

// dates render italicized wherever they appear
const TimeValue = styled.span`
  font-style: italic;
`;

// collapsed height cap (px) for description markdown, per variant — hover previews clamp tighter
const DESCRIPTION_COLLAPSED_PX: Record<SummaryVariant, number> = {
  [SummaryVariant.Compact]: 72,
  [SummaryVariant.Expanded]: 120,
};

// collapsed height cap (px) for code blocks (sigma rule / disassembly / decompilation), per variant
const CODE_COLLAPSED_PX: Record<SummaryVariant, number> = {
  [SummaryVariant.Compact]: 120,
  [SummaryVariant.Expanded]: 240,
};

// a monospace code block for raw content (sigma YAML, disassembly listing, decompiled source): preserves
// whitespace and scrolls horizontally for long lines rather than wrapping mid-token
const CodeBlock = styled.pre`
  margin: 4px 0 0;
  padding: 6px 8px;
  background: var(--thorium-secondary-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  font-family: monospace;
  font-size: 0.75rem;
  line-height: 1.35;
  white-space: pre;
  overflow-x: auto;
  max-width: 100%;
`;

const SectionHeading = styled(Subtitle)<{ $variant: SummaryVariant }>`
  margin-top: ${({ $variant }) => ($variant === SummaryVariant.Compact ? '6px' : '10px')};
`;

const Divider = styled.hr`
  margin: 8px 0;
  border-color: var(--thorium-panel-border);
  opacity: 0.6;
`;

const Row = styled.div<{ $variant: SummaryVariant }>`
  margin-top: ${({ $variant }) => ($variant === SummaryVariant.Compact ? '3px' : '6px')};
`;

// inline layout: the value flows as inline text right after the label. This is deliberately NOT flexbox
// — a flex value-column would indent wrapped lines to the column start; inline flow lets long values
// (paths, ids, reasoning) wrap back to the section's left edge, reclaiming that whitespace.
const InlineRow = styled.div<{ $variant: SummaryVariant }>`
  margin-top: ${({ $variant }) => ($variant === SummaryVariant.Compact ? '3px' : '6px')};
  word-break: break-all;
  overflow-wrap: anywhere;
`;

// inline label: same look as a `.subtitle`, but inline so the value continues on the same line
const InlineLabel = styled.span.attrs({ className: 'subtitle' })`
  margin-right: 6px;
`;

const TagRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-start;
  gap: 2px;
  margin-top: 4px;
`;

const NoteBanner = styled.div<{ $level: InfoNoteLevel }>`
  margin-top: 6px;
  padding: 3px 6px;
  font-size: 0.72rem;
  border-radius: 4px;
  background-color: ${({ $level }) =>
    $level === InfoNoteLevel.Warning ? 'var(--thorium-warning-secondary-bg)' : 'var(--thorium-info-secondary-bg)'};
  color: var(--thorium-text);
`;

/**
 * Decide whether a field renders inline (`Label: value`) or stacked (label above value).
 *
 * An explicit {@link InfoField.layout} always wins; otherwise it's derived from the value kind:
 * long prose (Markdown/Text) and multi-value arrays stack; times, numbers, and booleans go inline;
 * everything else (free-form strings, links, single-element arrays) stacks by default. Short strings
 * and short word-lists opt into inline via the explicit `layout` override in the adapters.
 *
 * @param field - The field to lay out.
 * @returns The resolved {@link FieldLayout}.
 */
export function resolveFieldLayout(field: InfoField): FieldLayout {
  if (field.layout) return field.layout;
  if (
    field.render === FieldRender.Markdown ||
    field.render === FieldRender.Text ||
    field.render === FieldRender.Path ||
    field.render === FieldRender.Code
  ) {
    return FieldLayout.Stacked;
  }
  if (Array.isArray(field.value) && field.value.length >= 2) return FieldLayout.Stacked;
  if (field.render === FieldRender.Time || typeof field.value === 'number' || typeof field.value === 'boolean') {
    return FieldLayout.Inline;
  }
  return FieldLayout.Stacked;
}

// render a field's value: named-entity link chips, a time value, a link, or a value badge
const FieldValue: React.FC<{ field: InfoField; variant: SummaryVariant }> = ({ field, variant }) => {
  if (field.links) {
    return (
      <>
        {field.links.map((link, i) =>
          link.href ? (
            <LinkChip key={i} to={link.href} target={field.external === false ? undefined : '_blank'}>
              {link.text}
            </LinkChip>
          ) : (
            <PlainChip key={i}>{link.text}</PlainChip>
          ),
        )}
      </>
    );
  }
  if (field.href) {
    return (
      <TitleLink to={field.href} target={field.external === false ? undefined : '_blank'}>
        {String(field.value)}
      </TitleLink>
    );
  }
  if (field.render === FieldRender.Markdown) {
    return <CollapsibleMarkdown collapsedMaxPx={DESCRIPTION_COLLAPSED_PX[variant]}>{String(field.value)}</CollapsibleMarkdown>;
  }
  if (field.render === FieldRender.Time) {
    return (
      <TimeValue>
        <Time>{String(field.value)}</Time>
      </TimeValue>
    );
  }
  if (field.render === FieldRender.Code) {
    return (
      <Collapsible maxPx={CODE_COLLAPSED_PX[variant]}>
        <CodeBlock>{String(field.value)}</CodeBlock>
      </Collapsible>
    );
  }
  if (field.render === FieldRender.Path) return <PathValue>{String(field.value)}</PathValue>;
  if (field.render === FieldRender.Text) return <TextValue>{String(field.value)}</TextValue>;
  return <FieldBadge color={field.danger ? 'DarkRed' : 'Gray'} noNull={true} field={field.value} />;
};

// render a single field row, inline (label + value flow as one wrapping paragraph) or stacked (label
// above value). Inline uses text flow (not flex) so wrapped value lines return to the section's left edge.
const FieldRow: React.FC<{ field: InfoField; variant: SummaryVariant }> = ({ field, variant }) => {
  if (resolveFieldLayout(field) === FieldLayout.Inline) {
    return (
      <InlineRow $variant={variant}>
        <InlineLabel>{field.label}</InlineLabel>
        <FieldValue field={field} variant={variant} />
      </InlineRow>
    );
  }
  return (
    <Row $variant={variant}>
      <Subtitle>{field.label}</Subtitle>
      <FieldValue field={field} variant={variant} />
    </Row>
  );
};

const SectionBlock: React.FC<{ section: InfoSection; tags?: Tags; variant: SummaryVariant; isLast: boolean; resource?: Entities }> = ({
  section,
  tags,
  variant,
  isLast,
  resource,
}) => (
  <>
    {/* compact drops section category titles and relies on the divider alone to delimit sections */}
    {section.heading && variant === SummaryVariant.Expanded && <SectionHeading $variant={variant}>{section.heading}</SectionHeading>}
    {section.fields?.map((field, idx) => (
      // include the index — a section can carry duplicate labels (e.g. `Type` from generic humanized fields)
      <FieldRow key={`${field.label}-${idx}`} field={field} variant={variant} />
    ))}
    {section.tagKeys && tags && (
      <TagRow>
        <FilteredNodeTags tags={tags} includeKeys={section.tagKeys} resource={resource} />
      </TagRow>
    )}
    {section.borderAfter && !isLast && <Divider />}
  </>
);

export interface EntitySummaryProps {
  model: InfoModel;
  /** How compactly to render; defaults to {@link SummaryVariant.Compact}. */
  variant?: SummaryVariant;
  /** Extra notes concatenated with `model.notes` (e.g. the tree's duplicate-node warning). */
  notes?: InfoNote[];
  /**
   * Labels / part tokens to hide. Each entry matches a field's `label` (case-insensitive) or a
   * `SummaryPart` token (kind/title/description/tags). Use to omit info already shown elsewhere — e.g.
   * a list that already renders the entity name passes `[SummaryPart.Title]`.
   */
  exclude?: string[];
  /**
   * Whether this node appears under multiple parents in the current tree. When provided, a compact
   * `Duplicate: <true/false>` badge renders in the header block (both variants).
   */
  duplicate?: boolean;
}

/**
 * Shared renderer for an entity / file / repo summary. Consumed by the graph side panel (expanded),
 * the graph node hover, the association tree hover, and the entity-name hovers (compact), fed by the
 * adapters in `info.ts`.
 */
const EntitySummary: React.FC<EntitySummaryProps> = ({ model, variant = SummaryVariant.Compact, notes, exclude, duplicate }) => {
  const visibility = applyExclusions(model, exclude);
  const { sections } = visibility;
  // the leading identifier block (ID / sha256 / provider) sits directly under the title; the rest follow
  // the description so the header reads name → id → description → details
  const [idSection, ...detailSections] = sections;
  // tag keys consumed by scoped sections are excluded from the catch-all tag block below
  const scopedTagKeys = sections.flatMap((s) => s.tagKeys ?? []);
  const tagKeys = model.tags ? Object.keys(model.tags) : [];
  const allNotes = [...(model.notes ?? []), ...(notes ?? [])];

  const description = visibility.description ? (
    model.description ? (
      <Row $variant={variant}>
        <Subtitle>Description</Subtitle>
        <Description>
          <CollapsibleMarkdown collapsedMaxPx={DESCRIPTION_COLLAPSED_PX[variant]}>{model.description}</CollapsibleMarkdown>
        </Description>
      </Row>
    ) : (
      // no description — show the title with an N/A tag so the absence is explicit
      <InlineRow $variant={variant}>
        <InlineLabel>Description</InlineLabel>
        <small>N/A</small>
      </InlineRow>
    )
  ) : null;

  return (
    <Wrapper $variant={variant}>
      {visibility.kind && <KindLabel>{model.kind}</KindLabel>}
      {visibility.title &&
        (model.titleHref ? (
          <TitleLink to={model.titleHref} target="_blank">
            {model.title}
          </TitleLink>
        ) : (
          <TitleText>{model.title}</TitleText>
        ))}
      {/* identifier block (ID/sha256) directly under the name — its own divider is deferred until after
          the description so name → id → description stay grouped in the header */}
      {idSection && (
        <SectionBlock
          section={{ ...idSection, borderAfter: false }}
          tags={model.tags}
          variant={variant}
          isLast={true}
          resource={model.resource}
        />
      )}
      {/* compact True/False badge (not a full-width note) so multi-parent nodes are flagged in the header */}
      {duplicate !== undefined && (
        <InlineRow $variant={variant}>
          <InlineLabel>Duplicate</InlineLabel>
          <FieldBadge color="Gray" noNull={false} field={duplicate} />
        </InlineRow>
      )}
      {description}
      {idSection?.borderAfter && detailSections.length > 0 && <Divider />}
      {detailSections.map((section, i) => (
        <SectionBlock
          key={section.heading ?? `section-${i}`}
          section={section}
          tags={model.tags}
          variant={variant}
          isLast={i === detailSections.length - 1}
          resource={model.resource}
        />
      ))}
      {visibility.tags && model.tags && (
        <Row $variant={variant}>
          <Subtitle>Tags</Subtitle>
          {tagKeys.length === 0 ? (
            <small>N/A</small>
          ) : (
            <TagRow>
              <FilteredNodeTags
                tags={model.tags}
                excludeKeys={scopedTagKeys.length > 0 ? scopedTagKeys : undefined}
                resource={model.resource}
              />
            </TagRow>
          )}
        </Row>
      )}
      {allNotes.map((note, i) => (
        <NoteBanner key={i} $level={note.level}>
          {note.message}
        </NoteBanner>
      ))}
    </Wrapper>
  );
};

export default EntitySummary;
