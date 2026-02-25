import React, { useState, useEffect } from 'react';
import { Alert, Card, Table } from 'react-bootstrap';
import sanitizeHtml from 'sanitize-html';

// project imports
import { getAlerts } from './alerts';
import ResultsFiles from './displays/files/ResultsFiles';
import ChildrenFiles from './displays/files/ChildrenFiles';
import '@styles/main.scss';
import { ResultRenderProps } from './props';
import { Value } from '@models/results';

const SafeHtml: React.FC<ResultRenderProps> = ({ result, sha256, tool }) => {
  const [errors, setErrors] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [resultsJson, setResultsJson] = useState<Value>([]);
  const [isJson, setIsJson] = useState(true);

  useEffect(() => {
    // set alerts and process results to json
    getAlerts(result.result, setResultsJson, setWarnings, setErrors, setIsJson, true);
  }, [result]);

  const SanitizeHTML = ({ html }) => <div dangerouslySetInnerHTML={{ __html: sanitizeHtml(html) }} />;

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
          <SanitizeHTML html={result.result} />
          <ResultsFiles result={result} sha256={sha256} tool={tool} />
          <ChildrenFiles result={result} sha256={sha256} tool={tool} />
        </Card.Body>
      </Card>
    </>
  );
};

export default SafeHtml;
