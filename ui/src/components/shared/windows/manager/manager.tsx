import React, { Fragment, JSX, useContext } from 'react';

// project imports
import { useWindowManagerProvider, WindowManagerProviderOptions } from './provider';
import { WindowContextType, WindowManagerContext } from './context';

/**
 * Window Manager component properties
 */
type WindowManagerProps = WindowManagerProviderOptions & {
  children: JSX.Element;
};

/**
 * Wrap application in a shared window manager provider
 */
export const WindowManager: React.FC<WindowManagerProps> = ({
  children,
  zRange = { start: 1000, end: 4000, step: 5 },
  canvasMargin = { top: 0, bottom: 0, start: 0, end: 0 },
  name,
}) => {
  const windows = useWindowManagerProvider({ zRange, canvasMargin, name });
  return (
    <WindowManagerContext.Provider value={windows}>
      {children}
      {/* render any managed windows here */}
      {windows.managedWindows.map((managedWindow) => (
        <Fragment key={managedWindow.id}>{managedWindow.window}</Fragment>
      ))}
    </WindowManagerContext.Provider>
  );
};

/**
 * Use hook for getting window manager context
 */
export const useWindowManager = (): WindowContextType => {
  const context = useContext(WindowManagerContext);
  if (context === undefined) {
    throw new Error('useWindowManager must be used within a WindowManager');
  }
  return context;
};
