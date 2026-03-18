import { useRef } from 'react';
import { Col, Row } from 'react-bootstrap';

const ChildrenFiles = ({ result, tool }) => {
  const childRef = useRef();
  if (result && result.children && Object.keys(result.children).length > 0) {
    return (
      <>
        <Row id={`children_${tool}`} ref={childRef} className="tool-results-text">
          <Col className="d-flex justify-content-center">
            <h5>Children</h5>
          </Col>
        </Row>
        {result.children &&
          Object.keys(result.children).map((child, idx) => (
            <Row key={`${child}_${idx}`} className="color-almost-white">
              <Col className="d-flex justify-content-center">
                <a href={'/file/' + child}>{child}</a>
              </Col>
            </Row>
          ))}
      </>
    );
  } else {
    return null;
  }
};

export default ChildrenFiles;
