import React from 'react';

// project imports
import { TitleProps } from './props';

const SimpleTitle: React.FC<TitleProps> = ({ children, className = '', ref }) => {
  return (
    <div ref={ref} className={`simple-title ${className ? className : ''}`}>
      {children}
    </div>
  );
};

export default SimpleTitle;
