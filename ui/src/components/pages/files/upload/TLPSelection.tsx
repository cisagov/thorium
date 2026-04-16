import React from 'react';
import { Button, Card } from 'react-bootstrap';
import { TLP_COLORS, TLPSelection as TLPSelectionState } from './types';

interface TLPSelectionProps {
  selectedTLP: TLPSelectionState;
  onTLPChange: (newSelection: TLPSelectionState) => void;
}

const TLPSelection: React.FC<TLPSelectionProps> = ({ selectedTLP, onTLPChange }) => {
  return (
    <Card className="panel">
      <Card.Body className="d-flex justify-content-center">
        {Object.keys(TLP_COLORS).map((tlp) => (
          <Button
            variant=""
            className={`tlp-btn ${TLP_COLORS[tlp]}-btn ${selectedTLP[tlp] ? 'selected' : ''}`}
            key={tlp}
            onClick={() => {
              const tempSelection: TLPSelectionState = {};
              Object.keys(TLP_COLORS).forEach((color) => {
                tempSelection[color] = color === tlp ? !selectedTLP[color] : false;
              });
              onTLPChange(tempSelection);
            }}
          >
            <b>{tlp}</b>
          </Button>
        ))}
      </Card.Body>
    </Card>
  );
};

export default TLPSelection;
