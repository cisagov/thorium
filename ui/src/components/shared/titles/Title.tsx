import React from 'react';

// project imports
import { TitleProps } from './props';

const Title: React.FC<TitleProps> = ({ children, className = '', small = false, ref }) => {
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

export default Title;
