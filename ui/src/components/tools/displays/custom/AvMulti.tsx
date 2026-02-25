import { Card, Row, Col } from 'react-bootstrap';

// project imports
import { ResultRenderProps } from '@components/tools/props';

const AvMulti: React.FC<ResultRenderProps> = ({ result }) => {
  const avResult = result.result != null && typeof result.result == 'object' ? result.result : {};

  return (
    <Card className="scroll-log tool-result">
      <Card.Body>
        <Row>
          <Col xs={4}>{'Timestamp:'}</Col>
          <Col>{result.uploaded}</Col>
        </Row>
        <Row>
          <Col xs={4}>{'Version:'}</Col>
          <Col>{result?.result != null ? result.result['Version'] : ''}</Col>
        </Row>
        <Row>
          <Col xs={4}>{'Result:'}</Col>
          <Col>{'Result' in avResult ? (avResult.Result as string) : 'Error'}</Col>
        </Row>
        {'Errors' in avResult && (
          <Row>
            <Col xs={4}>{'Errors:'}</Col>
            <Col>
              <pre>{avResult.Errors as string}</pre>
            </Col>
          </Row>
        )}
      </Card.Body>
    </Card>
  );
};

export default AvMulti;
