import { ResultRenderProps } from '@components/tools/props';
import React, { useRef } from 'react';
import { Col, Row } from 'react-bootstrap';

const ChildrenFiles: React.FC<ResultRenderProps> = ({ result, sha256, tool }) => {
  const childRef = useRef(null);
  if (result && result.children && Object.keys(result.children).length > 0) {
    return (
      <>
        <Row id={`children_${tool}`} ref={childRef}>
          <Col className="d-flex justify-content-center">
            <h5>Children</h5>
          </Col>
        </Row>
        {result.children &&
          Object.keys(result.children).map((child, idx) => (
            <Row key={`${child}_${idx}`}>
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
