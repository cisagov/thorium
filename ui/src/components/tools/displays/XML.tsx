import { useState, useEffect } from 'react';
import { Alert, Card } from 'react-bootstrap';
import XMLViewer from 'react-xml-viewer';

// project imports
import { getAlerts } from '../alerts';
import ResultsFiles from './files/ResultsFiles';
import ChildrenFiles from './files/ChildrenFiles';
import '@styles/main.scss';
import { ResultRenderProps } from '../props';
import { Value } from '@models/results';

const XML: React.FC<ResultRenderProps> = ({ result, sha256, tool }) => {
  const [errors, setErrors] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [resultsJson, setResultsJson] = useState<Value>({});
  const [isJson, setIsJson] = useState(true);

  useEffect(() => {
    // set alerts and process results to json
    getAlerts(result.result, setResultsJson, setWarnings, setErrors, setIsJson, true);
  }, [result]);

  // format string results or ignore result if json
  let parsedResult = '';
  // result is a string, replace new lines and format as such
  if (!isJson) {
    parsedResult = result?.result && typeof result.result === 'string' ? result.result.replace(/\\n/g, '\n').replace(/["]+/g, '') : '';
  } else {
    // ignore the results, they aren't strings
    if (JSON.stringify(resultsJson) == '{}') {
      parsedResult = '';
    } else {
      // there is non-empty json, display as string
      parsedResult = JSON.stringify(resultsJson);
    }
  }

  // Ocean theme from JSON tool renderer
  const thoriumTheme = {
    attributeKeyColor: '#96b5b4',
    attributeValueColor: '#d08770',
    tagColor: '#8fa1b3',
    textColor: '#a3be8c',
    separatorColor: 'tan',
  };

  return (
    <>
      <Card className="scroll-log tool-result">
        <Card.Body>
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
          <XMLViewer xml={parsedResult} theme={thoriumTheme} collapsible={true} initialCollapsedDepth={3} />
          <hr />
          <ResultsFiles result={result} sha256={sha256} tool={tool} />
          <ChildrenFiles result={result} sha256={sha256} tool={tool} />
        </Card.Body>
      </Card>
    </>
  );
};

export default XML;
