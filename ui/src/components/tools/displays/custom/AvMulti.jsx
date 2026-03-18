import { Card, Row, Col } from 'react-bootstrap';

const AvMulti = ({ result }) => {
  return (
    <Card className="scroll-log tool-result">
      <Card.Body>
        <Row>
          <Col xs={4}>{'Timestamp:'}</Col>
          <Col>{result.uploaded}</Col>
        </Row>
        <Row>
          <Col xs={4}>{'Version:'}</Col>
          <Col>{result.result.Version}</Col>
        </Row>
        <Row>
          <Col xs={4}>{'Result:'}</Col>
          <Col>{result.result.Result ? result.result.Result : 'Error'}</Col>
        </Row>
        {result.result['Errors'] && (
          <Row>
            <Col xs={4}>{'Errors:'}</Col>
            <Col>
              <pre>{result.result.Errors}</pre>
            </Col>
          </Row>
        )}
      </Card.Body>
    </Card>
  );
};

export default AvMulti;
