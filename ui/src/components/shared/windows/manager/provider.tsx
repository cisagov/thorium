import { createRef, ReactNode, RefObject, useCallback, useLayoutEffect, useRef, useState } from 'react';

// project imports
import { WindowData, ZRange } from './models';
import { Margin } from '../bounds';
import { sortIntegerStrings } from '@utilities';

export type WindowManagerProviderOptions = {
  zRange: ZRange;
  canvasMargin: Margin;
  name: string; // name of window manager for applications with multiple window managers
};

/*
 * Thorium window hooks for building out stateful window management
 */
export function useWindowManagerProvider({ zRange, canvasMargin, name }: WindowManagerProviderOptions) {
  const windows = useRef<WindowData[]>([]).current;
  const [managedWindows, setManagedWindows] = useState<WindowData[]>([]);
  const [updateCount, setUpdateCount] = useState(0);
  const managerState = useRef<{ nextId: number; zIndices: Record<number, number> }>({
    nextId: 0,
    zIndices: {},
  }).current;

  // Update
  useLayoutEffect(() => {
    windows.forEach((window) => {
      if (window.ref != null) {
        const element = window.ref.current;
        if (element) {
          const zIndex = managerState.zIndices[window.id];
          if (typeof zIndex === 'number') {
            element.style.zIndex = zIndex.toString();
          }
        }
      }
    });
  }, [updateCount]);

  const normalizeZIndexes = (zIndices: Record<number, number>) => {
    // build an indexMap based on the
    const indexMap: { [key: number]: number } = {};
    Object.keys(zIndices).forEach((id: string) => {
      indexMap[zIndices[Number(id)]] = Number(id);
    });
    let nextZIndex = zRange.start;
    const newZIndices: Record<number, number> = {};
    Object.keys(indexMap)
      .toSorted(sortIntegerStrings)
      .forEach((zIndex: string) => {
        newZIndices[Number(indexMap[Number(zIndex)])] = nextZIndex;
        nextZIndex += zRange.step;
      });
    return newZIndices;
  };

  /**
   * Manage rendering of a ReactNode window
   */
  const manageWindow = useCallback(
    <T extends HTMLElement>(windowId: string, window: ReactNode, ref: RefObject<T | null>) => {
      const windowRefIdx = windows.findIndex((window) => window.id === windowId);
      // window not yet registered or being managed
      if (windowRefIdx < 0 && managedWindows.findIndex((window) => window.id === windowId) < 0) {
        // register window
        windows.push({ id: windowId, ref, window: window });
        // add window to managed list
        setManagedWindows([...windows.filter((window) => window.window)]);
        // now we add to zIndices
        let currentMaxZ = Math.max(...Object.values(managerState.zIndices), zRange.start);
        // max index is above end of defined range, we must normalize and calculate zindex from normalized values
        if (currentMaxZ >= zRange.end) {
          const newZIndices = normalizeZIndexes({ ...managerState.zIndices, [windowId]: currentMaxZ + zRange.step });
          // update max zindex value within normalized window range
          currentMaxZ = Math.max(...Object.values(managerState.zIndices), zRange.start);
          // save normalized zindexes with new managed window included
          managerState.zIndices = newZIndices;
          // trigger update of zIndex by manager
          setUpdateCount((prev) => prev + 1);
        } else {
          managerState.zIndices = { ...managerState.zIndices, [windowId]: currentMaxZ + zRange.step };
          // trigger update of zIndex by manager
          setUpdateCount((prev) => prev + 1);
        }
        // window is registered, but not yet managed
      } else if (windowRefIdx >= 0 && managedWindows.findIndex((window) => window.id === windowId) < 0) {
        windows[windowRefIdx].window = window;
        setManagedWindows([...windows.filter((window) => (window.window ? true : false))]);
        // somehow window is managed but not registered, this is a bug
      } else if (windowRefIdx < 0 && managedWindows.findIndex((window) => window.id === windowId) > 0) {
        // TODO: need to check if in zIndexes
        console.log('Error: window already managed but somehow not registered.');
        windows.push({ id: windowId, ref, window: window });
        // window registered and managed already
      }
    },
    [windows, zRange, managerState],
  );

  /**
   * Create a new window ref and register it
   */
  const registerWindow = useCallback(
    <T extends HTMLElement>(node: T, id?: string): { ref: RefObject<T>; windowZRange: ZRange } => {
      let ref = createRef<T>() as RefObject<T>;
      ref.current = node;
      const windowId = id !== undefined ? id : `${name}-window-${managerState.nextId++}`;
      // check if id already exists in windows
      const existingRegisteredWindowIndex = windows.findIndex((window) => window.id === windowId);
      // window with ID already found registered
      if (existingRegisteredWindowIndex >= 0 && windows[existingRegisteredWindowIndex].ref) {
        // ref was null, so we will create it
        if (windows[existingRegisteredWindowIndex].ref?.current == null) {
          windows[existingRegisteredWindowIndex].ref = ref;
          // pass back existing ref
        } else {
          ref = windows[existingRegisteredWindowIndex].ref as RefObject<T>;
        }
        // new window registration since ID wasn't found
      } else {
        windows.push({ id: windowId, ref: ref });
      }
      // now we calculate window zRange including start/end of zIndex and hard code step to 1 since it can use every step in that range
      let windowZRange: ZRange = { start: zRange.start, end: zRange.end, step: 1 };
      // window already had a zINdex, we will use that for start
      if (managerState.zIndices[windowId]) {
        windowZRange = { start: managerState.zIndices[windowId], end: managerState.zIndices[windowId] + zRange.step, step: 1 };
        // window did not have a zIndex, lets build one that is above the highest current value
      } else {
        // new windows are placed at the front which is currentMaxZ + zRange.step
        let currentMaxZ = Math.max(...Object.values(managerState.zIndices), zRange.start);
        // window zRange fits within the step of the manager's range
        windowZRange = { start: currentMaxZ + zRange.step, end: currentMaxZ + zRange.step, step: 1 };
        if (currentMaxZ >= zRange.end) {
          const newZIndices = normalizeZIndexes({ ...managerState.zIndices, [windowId]: currentMaxZ + zRange.step });
          // update max zindex value within normalized window range
          currentMaxZ = Math.max(...Object.values(managerState.zIndices), zRange.start);
          windowZRange = { start: currentMaxZ + zRange.step, end: currentMaxZ + zRange.step, step: 1 };
          //managerState
          managerState.zIndices = newZIndices;
          setUpdateCount((prev) => prev + 1);
        } else {
          managerState.zIndices = { ...managerState.zIndices, [windowId]: currentMaxZ + zRange.step };
          setUpdateCount((prev) => prev + 1);
        }
      }
      return {
        ref: ref,
        windowZRange: windowZRange,
      };
    },
    [windows, zRange],
  );

  /**
   * Handle zIndex changes due to window button presses
   */
  const onWindowClick = useCallback(
    <T extends HTMLElement>(clickedRef: RefObject<T>) => {
      const window = windows.find((w) => w.ref === clickedRef);
      if (!window) return;
      // Clicked windows are brought to the front
      const currentMaxZ = Math.max(...Object.values(managerState.zIndices), zRange.start);
      // clicked window is already at top of stack
      if (currentMaxZ == managerState.zIndices[window.id]) {
        return;
      }
      let newZIndices = { ...managerState.zIndices, [window.id]: currentMaxZ + zRange.step };
      if (currentMaxZ >= zRange.end) {
        newZIndices = normalizeZIndexes({ ...managerState.zIndices, [window.id]: currentMaxZ + zRange.step });
      }
      managerState.zIndices = newZIndices;
      setUpdateCount((prev) => prev + 1);
    },
    [windows, zRange],
  );

  /**
   * Handle closing window and removing it from tracking
   */
  const onWindowClose = useCallback(
    <T extends HTMLElement>(clickedRef: RefObject<T>) => {
      const clickedWindow = windows.find((w) => w.ref === clickedRef);
      if (!clickedWindow) return;
      const updatedZIndices = { ...managerState.zIndices };
      delete updatedZIndices[clickedWindow.id];
      managerState.zIndices = updatedZIndices;
      const windowIndex = windows.findIndex((window) => window.id === clickedWindow.id);
      if (windowIndex >= 0) {
        windows.splice(windowIndex, 1);
        setManagedWindows(windows.filter((window) => window.id !== clickedWindow.id && window.window));
        const newZIndices = normalizeZIndexes(managerState.zIndices);
        managerState.zIndices = newZIndices;
        setUpdateCount((prev) => prev + 1);
      }
    },
    [windows, zRange],
  );

  return {
    zRange,
    canvasMargin,
    registerWindow,
    onWindowClick,
    onWindowClose,
    manageWindow,
    managedWindows,
  };
}
