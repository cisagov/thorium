import React from 'react';
import { Col, Form, Row } from 'react-bootstrap';
import Subtitle from '@components/shared/titles/Subtitle';

interface OriginFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: string;
  isInvalid?: boolean;
  feedback?: string;
}

const OriginField: React.FC<OriginFieldProps> = ({
  label,
  value,
  onChange,
  placeholder = 'optional',
  type = 'text',
  isInvalid,
  feedback,
}) => {
  return (
    <>
      <br />
      <Row>
        <Col className="name-width" xs={2}>
          <Subtitle>{label}</Subtitle>
        </Col>
        <Col xs={5}>
          <Form.Control
            type={type}
            placeholder={placeholder}
            value={value}
            onChange={(e) => onChange(type === 'text' ? String(e.target.value) : e.target.value)}
            isInvalid={isInvalid}
          />
          {feedback && <Form.Control.Feedback type="invalid">{feedback}</Form.Control.Feedback>}
        </Col>
      </Row>
    </>
  );
};

export default OriginField;
