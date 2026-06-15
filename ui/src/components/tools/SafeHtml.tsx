import DOMPurify from 'dompurify';
import React, { useState, useEffect } from 'react';
import { Card } from 'react-bootstrap';

// project imports
import { getAlerts } from './alerts';
import { ResultRenderProps } from './props';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import '@styles/main.scss';

// spec: ./ToolResult.spec.md

/**
 * Render a tool result's HTML output safely: extracts alerts via {@link getAlerts}, sanitizes the
 * HTML with DOMPurify, and injects it via dangerouslySetInnerHTML.
 */
const SafeHtml: React.FC<ResultRenderProps> = ({ result }) => {
  const [errors, setErrors] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  useEffect(() => {
    getAlerts(
      result.result,
      () => {},
      setWarnings,
      setErrors,
      () => {},
      true,
    );
  }, [result]);

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
          <div
            dangerouslySetInnerHTML={{
              __html: DOMPurify.sanitize(typeof result.result === 'string' ? result.result : ''),
            }}
          />
        </Card.Body>
      </Card>
    </>
  );
};

export default SafeHtml;
