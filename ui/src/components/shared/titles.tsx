import React from 'react';
import styled from 'styled-components';

interface TitleProps {
  children: React.ReactNode;
  className?: string;
  small?: boolean;
}

export const Subtitle: React.FC<TitleProps> = ({ children, className = '' }) => {
  return <div className={`subtitle ${className ? className : ''}`}>{children}</div>;
};

export const SimpleSubtitle: React.FC<TitleProps> = ({ children, className = '' }) => {
  return <div className={`simple-subtitle ${className ? className : ''}`}>{children}</div>;
};

export const Title: React.FC<TitleProps> = ({ children, className = '', small = false }) => {
  if (small) {
    return <div className={`small-title ${className ? className : ''}`}>{children}</div>;
  } else {
    return <div className={`title ${className ? className : ''}`}>{children}</div>;
  }
};

export const SimpleTitle: React.FC<TitleProps> = ({ children, className = '' }) => {
  return <div className={`simple-title ${className ? className : ''}`}>{children}</div>;
};

const BannerDiv = styled.div`
  color: var(--thorium-text);
  text-transform: uppercase;
  text-wrap: wrap;
  font-size: 1.5rem;
`;

export const Banner: React.FC<TitleProps> = ({ children, className = '' }) => {
  return <BannerDiv className={`${className}`}>{children}</BannerDiv>;
};
