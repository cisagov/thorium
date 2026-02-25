import { useState, useEffect } from 'react';
import { Alert, Card, Row } from 'react-bootstrap';

// project imports
import { getAlerts } from '../alerts';
import ResultsFiles from './files/ResultsFiles';
import ChildrenFiles from './files/ChildrenFiles';
import { ResultRenderProps } from '../props';
import { Value } from '@models/results';

type StringResultRenderProps = ResultRenderProps & {
  warnings: string[];
  errors: string[];
};

const String: React.FC<StringResultRenderProps> = ({ result, sha256, tool, warnings, errors }) => {
  const [parsedErrors, setParsedErrors] = useState<string[]>([]);
  const [parsedWarnings, setParsedWarnings] = useState<string[]>([]);
  const [resultsJson, setResultsJson] = useState<Value>({});
  const [isJson, setIsJson] = useState(true);

  // Check to see if this is json or string
  // it might be json in cases where results were too large to display
  // in which case an object w/ warning will be returned
  useEffect(() => {
    // set alerts and process results to json
    getAlerts(result.result, setResultsJson, setParsedWarnings, setParsedErrors, setIsJson, true);
    // combine any errors and warnings with those that are passed in to component
    errors.push(...parsedErrors);
    warnings.push(...parsedWarnings);
  }, [result]);

  // format string results or ignore result if json
  let newResult = '';
  // result is a string, replace new lines and format as such
  if (!isJson) {
    newResult = result?.result && typeof result.result === 'string' ? result.result.replace(/\\n/g, '\n').replace(/["]+/g, '') : '';
  } else {
    // ignore the results, they aren't strings
    if (JSON.stringify(resultsJson) == '{}') {
      newResult = '';
    } else {
      // there is non-empty json, display as string
      newResult = JSON.stringify(resultsJson);
    }
  }

  return (
    <Card className="scroll-log tool-result">
      <Row>
        {errors.map((err, idx) => (
          <center key={idx}>
            <Alert variant="danger">{err}</Alert>
          </center>
        ))}
        {warnings.map((warn, idx) => (
          <center key={idx}>
            <Alert variant="warning">{warn}</Alert>
          </center>
        ))}
      </Row>
      <Row>
        <pre>{newResult}</pre>
      </Row>
      <ResultsFiles result={result} sha256={sha256} tool={tool} />
      <ChildrenFiles result={result} sha256={sha256} tool={tool} />
    </Card>
  );
};

export default String;
