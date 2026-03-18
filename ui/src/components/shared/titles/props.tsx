import React, { RefObject } from 'react';

export interface TitleProps {
  children: React.ReactNode;
  className?: string;
  small?: boolean;
  ref?: RefObject<any>;
}
