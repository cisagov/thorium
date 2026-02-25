import { Alert, Card, Col, Row } from 'react-bootstrap';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';

// project imports
import ResultsFiles from './files/ResultsFiles';
import { ResultRenderProps } from '../props';
import { ChildrenFiles } from './files';

const MAX_LENGTH = 100000;

const Disassembly: React.FC<ResultRenderProps> = ({ result, sha256, tool }) => {
  const rawCodeString = result?.result && typeof result.result === 'string' ? result.result.replace(/\\n/g, '\n').replace(/["]+/g, '') : '';
  const totalCodeSize = rawCodeString.length;
  const codeString = rawCodeString.substring(0, MAX_LENGTH);
  // trigger warning if code was truncated due to large size
  let truncated = false;
  if (rawCodeString.length > MAX_LENGTH) {
    truncated = true;
  }
  return (
    <Card className="scroll-log tool-result">
      <ResultsFiles result={result} sha256={sha256} tool={tool} />
      {truncated ? (
        <Row>
          <center>
            <Alert variant="warning">
              {`The rendered disassembly has been truncated
                due to its large size: ${totalCodeSize} bytes`}
            </Alert>
          </center>
        </Row>
      ) : null}
      <SyntaxHighlighter style={atomOneDark}>{codeString}</SyntaxHighlighter>
      <hr />
      <ResultsFiles result={result} sha256={sha256} tool={tool} />
      <ChildrenFiles result={result} sha256={sha256} tool={tool} />
    </Card>
  );
};

export default Disassembly;
