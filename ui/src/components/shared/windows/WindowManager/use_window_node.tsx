import { ReactNode, RefObject, useCallback, useLayoutEffect, useRef, useState } from 'react';

// project imports
import { useWindowManager } from './use_window_manager';

// Per-window hook: registers with the WindowManager and returns refs + callbacks for z-index, click, and close
export function useWindowNode<N extends HTMLElement>(id?: string) {
  const { canvasMargin, registerWindow, onWindowClick, onWindowClose, zRange: managerZRange, manageWindow } = useWindowManager();
  const nodeRef = useRef<N | null>(null);
  const storedRef = useRef<RefObject<N> | null>(null);
  const [windowZRange, setWindowZRange] = useState({ start: managerZRange.start, end: managerZRange.start + managerZRange.step, step: 1 });
  const pendingManagedWindow = useRef<{ id: string; window: ReactNode } | null>(null);

  const windowRef = useCallback(
    (node: N | null) => {
      if (node) {
        nodeRef.current = node;
        const { ref: registerRef, windowZRange: registerZRange } = registerWindow(node, id);
        storedRef.current = registerRef;
        setWindowZRange(registerZRange);
      }
    },
    [registerWindow],
  );

  // Schedule a managed window to be registered on next layout effect
  const addManagedWindow = useCallback((windowId: string, window: ReactNode) => {
    pendingManagedWindow.current = { id: windowId, window };
  }, []);

  // Register pending managed windows during layout (hooks must be called at top level)
  useLayoutEffect(() => {
    if (pendingManagedWindow.current) {
      const { id: windowId, window } = pendingManagedWindow.current;
      manageWindow(windowId, window, nodeRef);
      pendingManagedWindow.current = null;
    }
  });

  const handleWindowClick = useCallback(() => {
    if (storedRef.current) onWindowClick(storedRef.current);
  }, [onWindowClick]);

  const handleWindowClose = useCallback(() => {
    if (storedRef.current) onWindowClose(storedRef.current);
  }, [onWindowClose]);

  return {
    windowRef,
    windowZRange,
    nodeRef,
    onWindowClick: handleWindowClick,
    onWindowClose: handleWindowClose,
    margin: canvasMargin,
    addManagedWindow,
  };
}

/*
  Example use of managed windows:

  const { addManagedWindow } = useWindowNode();

  const uploadRef = useRef(null);
  addManagedWindow(
    'upload-window-test1',
    <OverlayWindow
      id="upload-window-test1"
      parentRef={uploadRef}
      show={true}
      placement={Placement.Center}
      title="managed window"
      width={400}
      height={400}
    >
      This is a test
    </OverlayWindow>,
  );

  ...
  <div ref={windowRef}>...</div>
*/

/*
Example of using an unmanaged window:

  <OverlayWindow
    show={true}
    positioning={PositionType.Fixed}
    placement={Placement.BottomRight}
    title="unmanaged window"
    width={400}
    height={400}
  >
    This is a test
  </OverlayWindow>
*/
