import { useState, useEffect } from 'react';
import { Card } from 'react-bootstrap';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import { default as MarkdownHtml } from 'react-markdown';
import remarkGfm from 'remark-gfm';

// project imports
import { getAlerts } from '../alerts';
import ResultsFiles from './files/ResultsFiles';
import ChildrenFiles from './files/ChildrenFiles';
import { ResultRenderProps } from '../props';
import '@styles/main.scss';
import { Value } from '@models/results';

const Markdown: React.FC<ResultRenderProps> = ({ result, sha256, tool }) => {
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

  return (
    <>
      <Card className="scroll-log tool-result">
        <Card.Body>
          {errors.map((err, idx) => (
            <AlertBanner key={idx}>{err}</AlertBanner>
          ))}
          {warnings.map((warn, idx) => (
            <AlertBanner key={idx} severity={Severity.Warning}>
              {warn}
            </AlertBanner>
          ))}
          <center>
            <MarkdownHtml remarkPlugins={[remarkGfm]}>{parsedResult}</MarkdownHtml>
          </center>
          <hr />
          <ResultsFiles result={result} sha256={sha256} tool={tool} />
          <ChildrenFiles result={result} sha256={sha256} tool={tool} />
        </Card.Body>
      </Card>
    </>
  );
};

export default Markdown;
