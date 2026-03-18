import React from 'react';
import styled from 'styled-components';

// project imports
import { TitleProps } from './props';

const BannerDiv = styled.div`
  color: var(--thorium-text);
  text-transform: uppercase;
  text-wrap: wrap;
  font-size: 1.5rem;
`;

const Banner: React.FC<TitleProps> = ({ children, className = '', ref }) => {
  return (
    <BannerDiv ref={ref} className={`${className}`}>
      {children}
    </BannerDiv>
  );
};

export default Banner;
