import React, { RefObject } from 'react';
import styled from 'styled-components';

interface TitleProps {
  children: React.ReactNode;
  className?: string;
  small?: boolean;
  ref?: RefObject<any>;
}

export const Subtitle: React.FC<TitleProps> = ({ children, className = '', ref }) => {
  return (
    <div ref={ref} className={`subtitle ${className ? className : ''}`}>
      {children}
    </div>
  );
};

export const SimpleSubtitle: React.FC<TitleProps> = ({ children, className = '', ref }) => {
  return (
    <div ref={ref} className={`simple-subtitle ${className ? className : ''}`}>
      {children}
    </div>
  );
};

export const Title: React.FC<TitleProps> = ({ children, className = '', small = false, ref }) => {
  if (small) {
    return (
      <div ref={ref} className={`small-title ${className ? className : ''}`}>
        {children}
      </div>
    );
  } else {
    return (
      <div ref={ref} className={`title ${className ? className : ''}`}>
        {children}
      </div>
    );
  }
};

export const SimpleTitle: React.FC<TitleProps> = ({ children, className = '', ref }) => {
  return (
    <div ref={ref} className={`simple-title ${className ? className : ''}`}>
      {children}
    </div>
  );
};

const BannerDiv = styled.div`
  color: var(--thorium-text);
  text-transform: uppercase;
  text-wrap: wrap;
  font-size: 1.5rem;
`;

export const Banner: React.FC<TitleProps> = ({ children, className = '', ref }) => {
  return (
    <BannerDiv ref={ref} className={`${className}`}>
      {children}
    </BannerDiv>
  );
};
