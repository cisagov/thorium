import { createRef, ReactNode, RefObject, useCallback, useLayoutEffect, useRef, useState } from 'react';

// project imports
import { WindowData, ZRange } from './models';
import { Margin } from '../bounds';
import { sortIntegerStrings } from '@utilities/sorting';

// spec: ./WindowManager.spec.md

export type WindowManagerProviderOptions = {
  zRange: ZRange;
  canvasMargin: Margin;
  name: string;
};

// Core hook that manages z-index tracking, window registration, and managed window rendering
function useWindowManagerProvider({ zRange, canvasMargin, name }: WindowManagerProviderOptions) {
  const windows = useRef<WindowData[]>([]).current;
  const [managedWindows, setManagedWindows] = useState<WindowData[]>([]);
  const [updateCount, setUpdateCount] = useState(0);
  const managerState = useRef<{ nextId: number; zIndices: Record<string, number> }>({
    nextId: 0,
    zIndices: {},
  }).current;

  // Apply z-index values from managerState to actual DOM elements
  useLayoutEffect(() => {
    windows.forEach((window) => {
      if (window.ref !== null) {
        const element = window.ref?.current;
        if (element) {
          const zIndex = managerState.zIndices[window.id];
          if (typeof zIndex === 'number') {
            element.style.zIndex = zIndex.toString();
          }
        }
      }
    });
  }, [updateCount]);

  // Compress z-indices back into the configured range to prevent unbounded growth
  const normalizeZIndexes = (zIndices: Record<string, number>) => {
    const indexMap: Record<number, string> = {};
    for (const id of Object.keys(zIndices)) {
      indexMap[zIndices[id]] = id;
    }
    let nextZIndex = zRange.start;
    const newZIndices: Record<string, number> = {};
    Object.keys(indexMap)
      .toSorted(sortIntegerStrings)
      .forEach((zIndex: string) => {
        newZIndices[indexMap[Number(zIndex)]] = nextZIndex;
        nextZIndex += zRange.step;
      });
    return newZIndices;
  };

  // Register a ReactNode to be rendered by the WindowManager
  const manageWindow = useCallback(
    <T extends HTMLElement>(windowId: string, window: ReactNode, ref: RefObject<T | null>) => {
      const windowRefIdx = windows.findIndex((w) => w.id === windowId);
      if (windowRefIdx < 0 && managedWindows.findIndex((w) => w.id === windowId) < 0) {
        // New window: register and assign z-index
        windows.push({ id: windowId, ref, window: window });
        setManagedWindows([...windows.filter((w) => w.window != null)]);
        let currentMaxZ = Math.max(...Object.values(managerState.zIndices), zRange.start);
        if (currentMaxZ >= zRange.end) {
          const newZIndices = normalizeZIndexes({ ...managerState.zIndices, [windowId]: currentMaxZ + zRange.step });
          currentMaxZ = Math.max(...Object.values(managerState.zIndices), zRange.start);
          managerState.zIndices = newZIndices;
          setUpdateCount((prev) => prev + 1);
        } else {
          managerState.zIndices = { ...managerState.zIndices, [windowId]: currentMaxZ + zRange.step };
          setUpdateCount((prev) => prev + 1);
        }
      } else if (windowRefIdx >= 0 && managedWindows.findIndex((w) => w.id === windowId) < 0) {
        // Registered but not yet managed: attach the window ReactNode
        windows[windowRefIdx].window = window;
        setManagedWindows([...windows.filter((w) => w.window != null)]);
      } else if (windowRefIdx < 0 && managedWindows.findIndex((w) => w.id === windowId) >= 0) {
        // Managed but not registered — should not happen; re-register the ref to recover
        windows.push({ id: windowId, ref, window: window });
      }
    },
    [windows, zRange, managerState],
  );

  // Create a window ref and register it with z-index tracking
  const registerWindow = useCallback(
    <T extends HTMLElement>(node: T, id?: string): { ref: RefObject<T>; windowZRange: ZRange } => {
      let ref = createRef<T>() as RefObject<T>;
      ref.current = node;
      const windowId = id !== undefined ? id : `${name}-window-${managerState.nextId++}`;
      const existingRegisteredWindowIndex = windows.findIndex((w) => w.id === windowId);
      if (existingRegisteredWindowIndex >= 0 && windows[existingRegisteredWindowIndex].ref) {
        if (windows[existingRegisteredWindowIndex].ref?.current === null) {
          windows[existingRegisteredWindowIndex].ref = ref;
        } else {
          ref = windows[existingRegisteredWindowIndex].ref as RefObject<T>;
        }
      } else {
        windows.push({ id: windowId, ref: ref });
      }
      let windowZRange: ZRange = { start: zRange.start, end: zRange.end, step: 1 };
      if (managerState.zIndices[windowId]) {
        windowZRange = { start: managerState.zIndices[windowId], end: managerState.zIndices[windowId] + zRange.step, step: 1 };
      } else {
        let currentMaxZ = Math.max(...Object.values(managerState.zIndices), zRange.start);
        windowZRange = { start: currentMaxZ + zRange.step, end: currentMaxZ + zRange.step, step: 1 };
        if (currentMaxZ >= zRange.end) {
          const newZIndices = normalizeZIndexes({ ...managerState.zIndices, [windowId]: currentMaxZ + zRange.step });
          currentMaxZ = Math.max(...Object.values(managerState.zIndices), zRange.start);
          windowZRange = { start: currentMaxZ + zRange.step, end: currentMaxZ + zRange.step, step: 1 };
          managerState.zIndices = newZIndices;
          setUpdateCount((prev) => prev + 1);
        } else {
          managerState.zIndices = { ...managerState.zIndices, [windowId]: currentMaxZ + zRange.step };
          setUpdateCount((prev) => prev + 1);
        }
      }
      return { ref, windowZRange };
    },
    [windows, zRange],
  );

  // Bring clicked window to front by assigning it the highest z-index
  const onWindowClick = useCallback(
    <T extends HTMLElement>(clickedRef: RefObject<T>) => {
      const window = windows.find((w) => w.ref === clickedRef);
      if (!window) return;
      const currentMaxZ = Math.max(...Object.values(managerState.zIndices), zRange.start);
      if (currentMaxZ === managerState.zIndices[window.id]) return;
      let newZIndices = { ...managerState.zIndices, [window.id]: currentMaxZ + zRange.step };
      if (currentMaxZ >= zRange.end) {
        newZIndices = normalizeZIndexes({ ...managerState.zIndices, [window.id]: currentMaxZ + zRange.step });
      }
      managerState.zIndices = newZIndices;
      setUpdateCount((prev) => prev + 1);
    },
    [windows, zRange],
  );

  // Remove window from tracking and normalize remaining z-indices
  const onWindowClose = useCallback(
    <T extends HTMLElement>(clickedRef: RefObject<T>) => {
      const clickedWindow = windows.find((w) => w.ref === clickedRef);
      if (!clickedWindow) return;
      const updatedZIndices = { ...managerState.zIndices };
      delete updatedZIndices[clickedWindow.id];
      managerState.zIndices = updatedZIndices;
      const windowIndex = windows.findIndex((w) => w.id === clickedWindow.id);
      if (windowIndex >= 0) {
        windows.splice(windowIndex, 1);
        setManagedWindows(windows.filter((w) => w.id !== clickedWindow.id && w.window != null));
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

export default useWindowManagerProvider;
