import { default as ReactMarkdown } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import styled from 'styled-components';

const MarkdownWrapper = styled.div`
  /* scale headings down from the browser default so markdown titles (# H1, ## H2, ...) don't
     render huge in the small/inline spaces where descriptions live; sizes step down proportionally
     and margins are tightened to match */
  h1 {
    font-size: 1.4em;
    margin: 0.4em 0 0.3em;
  }
  h2 {
    font-size: 1.25em;
    margin: 0.4em 0 0.3em;
  }
  h3 {
    font-size: 1.12em;
    margin: 0.4em 0 0.3em;
  }
  h4 {
    font-size: 1em;
    margin: 0.4em 0 0.3em;
  }
  h5 {
    font-size: 0.9em;
    margin: 0.4em 0 0.3em;
  }
  h6 {
    font-size: 0.85em;
    margin: 0.4em 0 0.3em;
  }

  table {
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
  }

  th,
  td {
    padding: 0.5rem;
    overflow-wrap: anywhere;
    word-break: break-word;
  }
`;

interface MarkdownProps {
  /** The raw markdown source to render (GFM enabled). */
  children: string;
}

/**
 * Render a markdown string as themed HTML, with GitHub-flavored markdown (tables, task lists, etc.) and
 * headings/tables scaled down so titles read sensibly in the small, inline spaces where descriptions live.
 *
 * @param children - The raw markdown source to render.
 * @returns The rendered markdown block.
 */
const Markdown: React.FC<MarkdownProps> = ({ children }) => (
  <MarkdownWrapper>
    <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
  </MarkdownWrapper>
);

export default Markdown;
