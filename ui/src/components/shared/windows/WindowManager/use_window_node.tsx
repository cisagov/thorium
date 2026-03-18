import { ReactNode, RefObject, useCallback, useLayoutEffect, useRef, useState } from 'react';

// project imports
import { useWindowManager } from './use_window_manager';

/**
 * Get window reference and action callbacks for a ref node
 */
export function useWindowNode<N extends HTMLElement>(id?: string) {
  const { canvasMargin, registerWindow, onWindowClick, onWindowClose, zRange: managerZRange, manageWindow } = useWindowManager();
  const nodeRef = useRef<N | null>(null);
  // We build this wrapper because you can't create
  const storedRef = useRef<RefObject<N> | null>(null);
  const [windowZRange, setWindowZRange] = useState({ start: managerZRange.start, end: managerZRange.start + managerZRange.step, step: 1 });
  // Create and return a windowRef generation callback
  const windowRef = useCallback(
    (node: N | null) => {
      // handle creation of initial node ref
      if (node && storedRef.current === null) {
        nodeRef.current = node;
        // Register the reference with the window manager
        const { ref: registerRef, windowZRange: registerZRange } = registerWindow(node, id);
        storedRef.current = registerRef;
        // update ZRange for window
        setWindowZRange(registerZRange);
        // handle recreation of the node ref that already exists
      } else if (node && storedRef.current !== null) {
        nodeRef.current = node;
        //onWindowClose(storedRef.current);
        const { ref: registerRef, windowZRange: registerZRange } = registerWindow(node, id);
        storedRef.current = registerRef;
        // update ZRange for window
        setWindowZRange(registerZRange);
      }
    },
    [registerWindow],
  );

  // register and display a new managed window
  const addManagedWindow = useCallback(
    (id: string, window: ReactNode) => {
      useLayoutEffect(() => {
        manageWindow(id, window, nodeRef);
      }, []);
    },
    [manageWindow],
  );

  // Click window with function generated reference.
  const handleWindowClick = useCallback(() => {
    if (storedRef.current) onWindowClick(storedRef.current);
  }, [onWindowClick]);

  // Handle removing window from window manager after close
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
