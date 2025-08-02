import React from 'react';
import { Card } from 'react-bootstrap';

interface CardProps {
  children: React.ReactNode;
  className?: string; // custom className pass through
  panel?: boolean; // whether to treat card as a panel
}

const ThoriumCard: React.FC<CardProps> = ({ children, className = '', panel = false }) => {
  if (panel) {
    return <Card className={`panel ${className}`}>{children}</Card>;
  }
  return <Card className={`body ${className}`}>{children}</Card>;
};

export { ThoriumCard as Card };
