// spec: ./EntityBrowser.spec.md
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { FaEyeSlash } from 'react-icons/fa6';

// project imports
import { useEntityBrowser } from './EntityBrowserContext';
import {
  HiddenControl,
  HiddenMenu,
  HiddenMenuAction,
  HiddenMenuHeader,
  HiddenMenuItem,
  HiddenMenuLabel,
  ToggleChip,
} from './EntityBrowser.styled';

/**
 * "Hidden (n)" chip with a dropdown listing each hidden node for per-item unhide plus a clear-all action.
 * Reads the hidden-node set from the {@link useEntityBrowser} context, so both {@link BrowserToolbar} and a
 * dashboard omnibar strip can render the same undo path. Renders nothing when nothing is hidden.
 */
const HiddenNodesControl: React.FC = () => {
  const { hiddenNodes, unhideNode, unhideAll, labelForNode } = useEntityBrowser();
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const count = hiddenNodes.size;
  // close the dropdown on an outside click or Escape so it behaves like a standard menu
  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [open]);
  const onClear = useCallback(() => {
    unhideAll();
    setOpen(false);
  }, [unhideAll]);
  if (count === 0) return null;
  const ids = Array.from(hiddenNodes);
  return (
    <HiddenControl ref={containerRef}>
      <ToggleChip
        type="button"
        $active={open}
        data-testid="entity-browser-hidden"
        aria-haspopup="true"
        aria-expanded={open}
        aria-label={`${count} hidden ${count === 1 ? 'item' : 'items'} — open to unhide`}
        onClick={() => setOpen((v) => !v)}
      >
        <FaEyeSlash size={12} /> Hidden ({count})
      </ToggleChip>
      {open && (
        <HiddenMenu role="menu" data-testid="entity-browser-hidden-menu">
          <HiddenMenuHeader>
            <span>Hidden items</span>
            <HiddenMenuAction type="button" data-testid="entity-browser-hidden-clear" aria-label="Unhide all items" onClick={onClear}>
              Clear all
            </HiddenMenuAction>
          </HiddenMenuHeader>
          {ids.map((id) => {
            const label = labelForNode(id);
            return (
              <HiddenMenuItem key={id} role="menuitem">
                <HiddenMenuLabel title={label}>{label}</HiddenMenuLabel>
                <HiddenMenuAction type="button" aria-label={`Unhide ${label}`} onClick={() => unhideNode(id)}>
                  Unhide
                </HiddenMenuAction>
              </HiddenMenuItem>
            );
          })}
        </HiddenMenu>
      )}
    </HiddenControl>
  );
};

export default HiddenNodesControl;
