import React from 'react';
import { Col, Container, Row, Spinner } from 'react-bootstrap';

interface SpinnerProps {
  loading: boolean; // whether spinner is in view
}

export const LoadingSpinner: React.FC<SpinnerProps> = ({ loading }) => {
  return (
    <Container hidden={!loading}>
      <Row>
        <Col className="d-flex justify-content-center m-4">
          <Spinner animation="border" className="loading" />
        </Col>
      </Row>
    </Container>
  );
};
