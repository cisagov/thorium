import { ReactNode, RefObject } from 'react';

// spec: ./WindowManager.spec.md

// zRange allowed for an element or application
export type ZRange = {
  start: number;
  end: number;
  step: number;
};

// Info for a unique window
export type WindowData = {
  id: string;
  ref?: RefObject<HTMLElement | null>;
  window?: ReactNode;
};
