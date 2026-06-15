import { Card } from 'react-bootstrap';
import { default as MarkdownHtml } from 'react-markdown';
import remarkGfm from 'remark-gfm';

// project imports
import ResultAlerts from './ResultAlerts';
import { useResultAlerts } from './useResultAlerts';
import { formatResultBody } from '../alerts';
import { ResultRenderProps } from '../props';
import '@styles/main.scss';

// spec: ../ToolResult.spec.md

/** Render a tool result as GitHub-flavored Markdown. */
const Markdown: React.FC<ResultRenderProps> = ({ result }) => {
  const { errors, warnings, resultsJson, isJson } = useResultAlerts(result.result, true);
  const parsedResult = formatResultBody(result.result, isJson, resultsJson);

  return (
    <>
      <Card className="scroll-log tool-result">
        <Card.Body>
          <ResultAlerts errors={errors} warnings={warnings} />
          <center>
            <MarkdownHtml remarkPlugins={[remarkGfm]}>{parsedResult}</MarkdownHtml>
          </center>
        </Card.Body>
      </Card>
    </>
  );
};

export default Markdown;
