import React from 'react';

// project imports
import { TitleProps } from './props';

// spec: ./titles.spec.md

const SimpleSubtitle: React.FC<TitleProps> = ({ children, className = '', ref }) => {
  return (
    <div ref={ref} className={`simple-subtitle ${className ? className : ''}`}>
      {children}
    </div>
  );
};

export default SimpleSubtitle;
