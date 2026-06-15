import { Card, Row } from 'react-bootstrap';

// project imports
import ResultAlerts from './ResultAlerts';
import { useResultAlerts } from './useResultAlerts';
import { formatResultBody } from '../alerts';
import { ResultRenderProps } from '../props';

// spec: ../ToolResult.spec.md

type StringResultRenderProps = ResultRenderProps & {
  warnings: string[];
  errors: string[];
};

/**
 * Render a tool result as preformatted text. It may still be JSON when a result was too large to
 * display (an object with a warning is returned), so alerts are parsed here in addition to any
 * passed in by the caller.
 */
const String: React.FC<StringResultRenderProps> = ({ result, warnings, errors }) => {
  const { errors: parsedErrors, warnings: parsedWarnings, resultsJson, isJson } = useResultAlerts(result.result, true);

  // show the alerts passed in by the caller alongside any parsed from this result; derive these
  // during render so they reflect the latest parsed state without mutating the props arrays
  const allErrors = [...errors, ...parsedErrors];
  const allWarnings = [...warnings, ...parsedWarnings];
  const newResult = formatResultBody(result.result, isJson, resultsJson);

  return (
    <Card className="scroll-log tool-result">
      <Row>
        <ResultAlerts errors={allErrors} warnings={allWarnings} />
      </Row>
      <Row>
        <pre>{newResult}</pre>
      </Row>
    </Card>
  );
};

export default String;
