import React from 'react';
import { Card, Col, Row } from 'react-bootstrap';
import { JSONTree } from 'react-json-tree';

// project imports
import ResultAlerts from './ResultAlerts';
import String from './String';
import { useResultAlerts } from './useResultAlerts';
import { ResultRenderProps } from '../props';
import { OceanJsonTheme } from '@components/shared/renderers/jsonTheme';

// spec: ../ToolResult.spec.md

/** Generic JSON result display backed by react-json-tree; falls back to the String display for non-JSON results. */
const JSON: React.FC<ResultRenderProps> = ({ result, sha256, tool }) => {
  const { errors, warnings, resultsJson, isJson } = useResultAlerts(result.result, false);
  // invert the dark token palette on light-background themes so the tree stays legible
  // const invertTheme = useJsonTreeInvert();

  return (
    <>
      {isJson ? (
        <Card className="scroll-log tool-result">
          <Row>
            <ResultAlerts errors={errors} warnings={warnings} />
          </Row>
          <Row>
            <Col>
              <JSONTree data={resultsJson} shouldExpandNodeInitially={() => false} hideRoot={true} theme={OceanJsonTheme} />
            </Col>
          </Row>
        </Card>
      ) : (
        <String result={result} sha256={sha256} tool={tool} warnings={warnings} errors={errors} />
      )}
    </>
  );
};

export default JSON;
