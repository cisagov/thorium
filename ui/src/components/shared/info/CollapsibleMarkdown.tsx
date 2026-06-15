import React from 'react';
import styled from 'styled-components';

// project imports
import Collapsible from './Collapsible';
import Markdown from '@components/shared/syntax/Markdown';

// spec: ./SPEC.md

// tightens markdown block spacing for the summary preview context (applied to the Collapsible clip)
const MarkdownClip = styled.div`
  p,
  ul,
  ol,
  pre,
  blockquote {
    margin: 0 0 0.4em;
  }
  p:last-child,
  ul:last-child,
  ol:last-child,
  pre:last-child {
    margin-bottom: 0;
  }
  ul,
  ol {
    padding-left: 1.2em;
  }
`;

interface CollapsibleMarkdownProps {
  /** The raw text; rendered as markdown (plain text formats fine, markdown syntax is honored). */
  children: string;
  /** Collapsed height cap in px; longer content is clipped behind an expand toggle. */
  collapsedMaxPx: number;
}

/**
 * Render (possibly long) text as markdown, collapsed to `collapsedMaxPx` by default with a Show
 * more/less toggle when the content overflows. Used for view-only description previews so a long
 * description never dominates the summary until the user opts to expand it.
 *
 * Thin wrapper over the shared {@link Collapsible} (which owns the collapse/fade/overflow-measure
 * pattern); this component only supplies the markdown renderer and its block-spacing tweaks.
 */
const CollapsibleMarkdown: React.FC<CollapsibleMarkdownProps> = ({ children, collapsedMaxPx }) => (
  <Collapsible maxPx={collapsedMaxPx}>
    <MarkdownClip>
      <Markdown>{children}</Markdown>
    </MarkdownClip>
  </Collapsible>
);

export default CollapsibleMarkdown;
