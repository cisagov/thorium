import { default as ReactMarkdown } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import styled from 'styled-components';

const MarkdownWrapper = styled.div`
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
  children: string;
}

const Markdown: React.FC<MarkdownProps> = ({ children }) => (
  <MarkdownWrapper>
    <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
  </MarkdownWrapper>
);

export default Markdown;
