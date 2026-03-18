import React, { useState, useEffect } from 'react';
import { Alert, Card, Col, Row } from 'react-bootstrap';
import { JSONTree } from 'react-json-tree';

// project imports
import String from './String';
import { getAlerts } from '../alerts';
import ResultsFiles from './files/ResultsFiles';
import ChildrenFiles from './files/ChildrenFiles';

const OceanJsonTheme = {
  scheme: 'Ocean',
  author: 'Chris Kempson (http://chriskempson.com)',
  // this value is pulled from styles/colors.scss
  base00: 'var(--thorium-panel-color)',
  base01: '#343d46',
  base02: '#4f5b66',
  base03: '#65737e',
  base04: '#a7adba',
  base05: '#c0c5ce',
  base06: '#dfe1e8',
  base07: '#eff1f5',
  base08: '#bf616a',
  base09: '#d08770',
  base0A: '#ebcb8b',
  base0B: '#a3be8c',
  base0C: '#96b5b4',
  base0D: '#8fa1b3',
  base0E: '#b48ead',
  base0F: '#ab7967',
};

// generic json dump using react-json-view library
const JSON = ({ result, sha256, tool }) => {
  const [errors, setErrors] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [resultsJson, setResultsJson] = useState({});
  const [isJson, setIsJson] = useState(true);

  useEffect(() => {
    // set alerts and process results to json
    getAlerts(result.result, setResultsJson, setWarnings, setErrors, setIsJson);
  }, [result]);

  return (
    <>
      {isJson ? (
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
          {Object.keys(resultsJson).length > 0 && (
            <Row>
              <Col>
                <JSONTree
                  data={resultsJson}
                  shouldExpandNodeInitially={() => true}
                  hideRoot={true}
                  theme={OceanJsonTheme}
                  invertTheme={false}
                />
              </Col>
            </Row>
          )}
          <ResultsFiles result={result} sha256={sha256} tool={tool} />
          <ChildrenFiles result={result} tool={tool} />
        </Card>
      ) : (
        <String result={result} warnings={warnings} errors={errors} />
      )}
    </>
  );
};

export { JSON as default, OceanJsonTheme };
