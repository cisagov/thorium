import { Alert, Card, Col, Row } from 'react-bootstrap';

const TC2 = ({ result }) => {
  // return a list of links to children files, one file per row
  if (result && result.result && result.result['children'] && Object.keys(result.result.children).length > 0) {
    return (
      <Card className="scroll-log tool-result">
        <Card.Body>
          <Row>
            <Col className="d-flex justify-content-center">
              <h5>Unpacked Children</h5>
            </Col>
          </Row>
          <br />
          {Object.values(result.result.children).map((sha256, idx) => (
            <Row key={idx}>
              <Col className="d-flex justify-content-center">
                <a className="highlight-dark" href={'/file/' + sha256}>
                  {sha256}
                </a>
              </Col>
            </Row>
          ))}
        </Card.Body>
      </Card>
    );
    // no children were found
  } else {
    return (
      <Card className="scroll-log tool-result">
        <Card.Body>
          <Alert variant="info" className="d-flex justify-content-center">
            No children unpacked
          </Alert>
        </Card.Body>
      </Card>
    );
  }
};

export default TC2;
