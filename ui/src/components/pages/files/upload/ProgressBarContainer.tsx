import React from 'react';
import { ProgressBar } from 'react-bootstrap';

// spec: ./upload.spec.md

interface ProgressBarProps {
  name: string;
  value: number;
  error: boolean;
}

const ProgressBarContainer: React.FC<ProgressBarProps> = ({ name, value, error }) => {
  return (
    <>
      {value < 100 && <ProgressBar animated key={name} label={value} now={value} className={error ? 'warning-bar' : 'info-bar'} />}
      {value >= 100 && <ProgressBar key={name} label={value} now={value} className={error ? 'danger-bar' : 'success-bar'} />}
    </>
  );
};

export default ProgressBarContainer;
