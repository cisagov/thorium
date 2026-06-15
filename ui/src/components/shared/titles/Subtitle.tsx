import React from 'react';

// project imports
import { TitleProps } from './props';

// spec: ./titles.spec.md

const Subtitle: React.FC<TitleProps> = ({ children, className = '', ref }) => {
  return (
    <div ref={ref} className={`subtitle ${className ? className : ''}`}>
      {children}
    </div>
  );
};

export default Subtitle;
