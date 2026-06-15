import { Card, Row } from 'react-bootstrap';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs';

// project imports
import { normalizeResultText } from '../alerts';
import { ResultRenderProps } from '../props';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';

// spec: ../ToolResult.spec.md

// results larger than this are truncated before highlighting to keep rendering responsive
const MAX_LENGTH = 100000;

/** Render a tool result as syntax-highlighted disassembly, truncating oversized output with a warning. */
const Disassembly: React.FC<ResultRenderProps> = ({ result }) => {
  const rawCodeString = result?.result && typeof result.result === 'string' ? normalizeResultText(result.result) : '';
  const totalCodeSize = rawCodeString.length;
  const codeString = rawCodeString.substring(0, MAX_LENGTH);
  const truncated = rawCodeString.length > MAX_LENGTH;
  return (
    <Card className="scroll-log tool-result">
      {truncated ? (
        <Row>
          <AlertBanner severity={Severity.Warning}>
            {`The rendered disassembly has been truncated
                due to its large size: ${totalCodeSize} bytes`}
          </AlertBanner>
        </Row>
      ) : null}
      {/* eslint-disable-next-line @typescript-eslint/no-unsafe-assignment -- atomOneDark has mismatched types */}
      <SyntaxHighlighter style={atomOneDark}>{codeString}</SyntaxHighlighter>
    </Card>
  );
};

export default Disassembly;
