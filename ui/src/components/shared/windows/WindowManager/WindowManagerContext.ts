import { createContext, ReactNode, RefObject } from 'react';

// project imports
import { Margin } from '../bounds';
import { ZRange } from './models';

type WindowContextType = {
  zRange: ZRange;
  canvasMargin: Margin;
  registerWindow: <T extends HTMLElement>(instance: T, id?: string) => { ref: RefObject<T>; windowZRange: ZRange };
  onWindowClick: <T extends HTMLElement>(windowRef: RefObject<T>) => void;
  onWindowClose: <T extends HTMLElement>(windowRef: RefObject<T>) => void;
  manageWindow: <T extends HTMLElement>(id: string, window: ReactNode, windowRef: RefObject<T | null>) => void;
};

export const WindowManagerContext = createContext<WindowContextType | undefined>(undefined);
