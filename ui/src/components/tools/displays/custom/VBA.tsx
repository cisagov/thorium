import { Fragment } from 'react';
import { Row, Col, Card } from 'react-bootstrap';

// project imports
import { Output, Value } from '@models/results';

// spec: ../../ToolResult.spec.md

interface VBAProps {
  result: Output;
}

/**
 * Custom display for VBA macro extraction: shows the scan timestamp and each top-level scalar field
 * of the structured result, omitting the large `analysis`, `form_strings`, and `macros` sections.
 */
const VBA: React.FC<VBAProps> = ({ result }) => {
  const newresult = result.result as { [key: string]: Value };
  return (
    <Fragment>
      <Card className="scroll-log tool-result">
        <Card.Body>
          <Row>
            <Col xs={2}> {'Timestamp:'}</Col>
            <Col>{result.uploaded}</Col>
          </Row>
          {Object.keys(newresult).map((key) => {
            if (key != 'analysis' && key != 'form_strings' && key != 'macros') {
              return (
                <Row key={key}>
                  <Col xs={2}>{key.charAt(0).toUpperCase() + key.slice(1)} :</Col>
                  <Col>{result && JSON.stringify(newresult[key]).slice(1, -1)}</Col>
                </Row>
              );
            }
          })}
        </Card.Body>
      </Card>
    </Fragment>
  );
};

export default VBA;
