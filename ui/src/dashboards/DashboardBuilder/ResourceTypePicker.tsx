// spec: ./SPEC.md

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { FaCheck, FaChevronDown } from 'react-icons/fa6';
import styled from 'styled-components';

// project imports
import { BROWSABLE_KINDS, BrowseMode, TAG_MODE } from './types';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import { spacers } from '@styles';
import { entityLabel } from '@models/entities/entities';

/**
 * The positioned wrapper anchoring the options menu directly beneath the trigger.
 *
 * `position: relative` scopes the absolutely-positioned {@link PickerMenu}; the min-width keeps the
 * control from collapsing narrower than the omnibar-styled trigger when its label is short.
 */
const PickerContainer = styled.div`
  position: relative;
  min-width: 200px;
`;

/**
 * The dropdown trigger, styled to mirror the omnibar entry field.
 *
 * Uses the shared `--thorium-omnibar-*` tokens (border/background) plus the same 5px radius and 40px
 * height so the picker reads as part of the same control family as the adjacent filters omnibar. While
 * open, the bottom corners are squared off so the trigger and menu join as one seam — matching the
 * omnibar's own entry-field-to-dropdown join.
 */
const PickerTrigger = styled.button<{ $open: boolean }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: ${spacers.three};
  width: 100%;
  height: 40px;
  padding: 0 ${spacers.four};
  border: 1px solid var(--thorium-omnibar-border);
  border-radius: 5px;
  background-color: var(--thorium-omnibar-bg);
  color: var(--thorium-text);
  font-size: 0.9rem;
  cursor: pointer;
  ${({ $open }) => $open && 'border-bottom-left-radius: 0; border-bottom-right-radius: 0;'}
`;

/**
 * The trigger's chevron affordance, flipped while the menu is open.
 *
 * A muted color and a short rotation transition so the open/closed state reads at a glance without
 * competing with the label.
 */
const Chevron = styled(FaChevronDown)<{ $open: boolean }>`
  flex-shrink: 0;
  color: var(--thorium-secondary-text);
  transition: transform 0.15s ease;
  ${({ $open }) => $open && 'transform: rotate(180deg);'}
`;

/**
 * The open options menu, mirroring the omnibar dropdown.
 *
 * Same omnibar tokens, seamless (top-borderless) join to the trigger, drop shadow, and scroll cap as
 * {@link OmnibarDropdown} so the two dropdowns look identical when open.
 */
const PickerMenu = styled.div`
  position: absolute;
  top: 100%;
  left: 0;
  width: 100%;
  border: 1px solid var(--thorium-omnibar-border);
  border-top: none;
  background-color: var(--thorium-omnibar-bg);
  box-shadow: 0 12px 14px var(--thorium-omnibar-dropdown-shadow);
  max-height: 400px;
  overflow-y: auto;
  z-index: 1000;
`;

/**
 * A single option row, highlighting on keyboard or pointer focus like an omnibar dropdown option and
 * marking the current selection with a checkmark.
 */
const PickerOption = styled.div<{ $focused: boolean }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: ${spacers.four};
  padding: 5px ${spacers.four};
  cursor: pointer;
  color: var(--thorium-text);
  background-color: ${({ $focused }) => ($focused ? 'var(--thorium-omnibar-dropdown-highlight)' : 'inherit')};
`;

/**
 * A resource-type option: the browse-mode value and its display label.
 */
interface PickerOptionItem {
  /// The browse mode this option selects (an entity kind or Tag mode).
  value: BrowseMode;
  /// The human-readable label shown for the option.
  label: string;
}

/**
 * Props for {@link ResourceTypePicker}.
 */
interface ResourceTypePickerProps {
  /// The currently selected browse mode (an entity kind or Tag mode).
  mode: BrowseMode;
  /// Called with the new mode when the user picks a different type.
  onChange: (mode: BrowseMode) => void;
}

/**
 * The dropdown that chooses which resource type to browse (File, Repo, entity kinds) or Tag mode.
 *
 * A bespoke dropdown (rather than a native `<select>`) so the trigger and menu can match the omnibar's
 * styling exactly — the picker sits inline in front of the filters omnibar and now reads as the same
 * control family. Selecting an entity kind drives the config-driven browse list; selecting Tag mode
 * swaps the list for a `TagSelect`-based key/value entry. Options come from {@link BROWSABLE_KINDS}
 * (stable order) with Tag mode appended.
 *
 * Keyboard: the trigger opens the menu on Enter/Space/ArrowDown, arrows move the focused option, Enter
 * selects it, and Escape closes. A pointer-down outside the control also closes it.
 *
 * @param props - See {@link ResourceTypePickerProps}.
 * @returns The resource-type picker control.
 */
const ResourceTypePicker: React.FC<ResourceTypePickerProps> = ({ mode, onChange }) => {
  // the fixed option set: every browsable entity kind (stable order) then Tag mode
  const options = useMemo<PickerOptionItem[]>(
    () => [...BROWSABLE_KINDS.map((kind) => ({ value: kind, label: entityLabel(kind) })), { value: TAG_MODE, label: 'Tag' }],
    [],
  );
  const [open, setOpen] = useState(false);
  // the option currently highlighted by keyboard/pointer while the menu is open
  const [focusIndex, setFocusIndex] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  // mode is always one of the options, so the label lookup should always resolve
  const currentLabel = options.find((option) => option.value === mode)?.label ?? '';
  // open the menu with the current selection pre-focused so keyboard nav starts from where the user is
  const openMenu = () => {
    const idx = options.findIndex((option) => option.value === mode);
    setFocusIndex(idx >= 0 ? idx : 0);
    setOpen(true);
  };
  // commit a selection and close the menu
  const select = (value: BrowseMode) => {
    onChange(value);
    setOpen(false);
  };
  // close on any pointer-down outside the control (the trigger's own click toggles it separately)
  useEffect(() => {
    if (!open) {
      return;
    }
    const handlePointerDown = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handlePointerDown);
    return () => document.removeEventListener('mousedown', handlePointerDown);
  }, [open]);
  // drive open/close and option navigation from the keyboard while the trigger holds focus
  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    switch (event.key) {
      case 'ArrowDown':
        // open the menu, or step the focus down within it
        event.preventDefault();
        if (open) {
          setFocusIndex((idx) => Math.min(idx + 1, options.length - 1));
        } else {
          openMenu();
        }
        break;
      case 'ArrowUp':
        // step the focus up within an open menu
        if (open) {
          event.preventDefault();
          setFocusIndex((idx) => Math.max(idx - 1, 0));
        }
        break;
      case 'Enter':
      case ' ':
        // select the focused option, or open the menu when closed
        event.preventDefault();
        if (open) {
          select(options[focusIndex].value);
        } else {
          openMenu();
        }
        break;
      case 'Escape':
        // dismiss an open menu without changing the selection
        if (open) {
          event.preventDefault();
          setOpen(false);
        }
        break;
    }
  };
  return (
    <PickerContainer ref={containerRef}>
      <OverlayTipRight
        tip="Choose what to browse and add to the dashboard. Pick Tag to seed by a tag key/value instead of a specific item."
        block
      >
        <PickerTrigger
          type="button"
          $open={open}
          aria-haspopup="listbox"
          aria-expanded={open}
          aria-label="Resource type to browse"
          onClick={() => (open ? setOpen(false) : openMenu())}
          onKeyDown={handleKeyDown}
        >
          <span>{currentLabel}</span>
          <Chevron $open={open} size={12} aria-hidden />
        </PickerTrigger>
      </OverlayTipRight>
      {open && (
        <PickerMenu role="listbox" aria-label="Resource type to browse">
          {options.map((option, idx) => (
            <PickerOption
              key={option.value}
              role="option"
              aria-selected={option.value === mode}
              $focused={idx === focusIndex}
              onMouseMove={() => setFocusIndex(idx)}
              onClick={() => select(option.value)}
            >
              <span>{option.label}</span>
              {option.value === mode && <FaCheck size={12} />}
            </PickerOption>
          ))}
        </PickerMenu>
      )}
    </PickerContainer>
  );
};

export default ResourceTypePicker;
