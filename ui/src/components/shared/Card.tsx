import React from 'react';
import { Card } from 'react-bootstrap';

// spec: ./Card.spec.md

interface CardProps {
  children: React.ReactNode;
  className?: string;
  /** Renders the card with panel styling instead of the default body styling. */
  panel?: boolean;
}

const ThoriumCard: React.FC<CardProps> = ({ children, className = '', panel = false }) => {
  if (panel) {
    return <Card className={`panel ${className}`}>{children}</Card>;
  }
  return <Card className={`body ${className}`}>{children}</Card>;
};

export { ThoriumCard as Card };
