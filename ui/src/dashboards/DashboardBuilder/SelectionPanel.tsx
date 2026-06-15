// spec: ./SPEC.md

import React, { useMemo } from 'react';

// project imports
import { selectionKey } from './builderReducer';
import { BuilderSelection } from './types';
import { SectionLabel } from './styles';
import SelectInputArray, { SelectInputArrayComponents } from '@components/shared/inputs/selectable/SelectInputArray';
import { OverlayTipTop } from '@components/shared/overlay/tips';
import type { ValueMap } from '@models/shared';

/**
 * A `components` override that hides the react-select menu entirely.
 *
 * When there are no removed items there is nothing to re-add, so the panel suppresses the (empty)
 * dropdown menu. Defined at module scope so its identity is stable across renders.
 */
const HIDDEN_MENU: SelectInputArrayComponents = { Menu: () => null };

/**
 * Props for {@link SelectionPanel}.
 */
interface SelectionPanelProps {
  /// The currently selected resources, rendered as chips.
  selected: BuilderSelection[];
  /// Previously removed resources, offered as the only re-add options.
  removed: BuilderSelection[];
  /// Called with the identity key of a chip the user removed (dispatches Remove).
  onRemove: (key: string) => void;
  /// Called with the identity key of an option the user re-added (dispatches Readd).
  onReadd: (key: string) => void;
}

/**
 * The running selection list, rendered with the non-typeable {@link SelectInputArray}.
 *
 * The chips are the current selections keyed by identity; the only dropdown options are previously
 * removed items, so the sole thing the dropdown offers is re-adding an accidental removal (never the
 * whole catalog). The menu is hidden when nothing has been removed. `valuesMap` maps each identity
 * key to its human-readable label so chips read as e.g. `malware.exe` rather than a raw sha256, and
 * refreshes when labels resolve after ids.
 *
 * Chip removal and option re-add are distinguished by comparing the new value set against the old:
 * a value that disappeared was removed; a value that appeared came from the removed options.
 *
 * @param props - See {@link SelectionPanelProps}.
 * @returns The selection panel section.
 */
const SelectionPanel: React.FC<SelectionPanelProps> = ({ selected, removed, onRemove, onReadd }) => {
  // the chip values are selection identity keys, so removes/re-adds map straight back to reducer keys
  const selectedKeys = useMemo(() => selected.map(selectionKey), [selected]);
  // options offered are ONLY removed items (re-add undo), keyed the same way as the chips
  const removedKeys = useMemo(() => removed.map(selectionKey), [removed]);
  // one id->label map covering both selected and removed so chips and options both render labels
  const valuesMap = useMemo<ValueMap>(() => {
    const map: ValueMap = {};
    for (const selection of [...selected, ...removed]) {
      map[selectionKey(selection)] = selection.label;
    }
    return map;
  }, [selected, removed]);

  // translate react-select's full-value-array change into a single Remove or Readd reducer action
  const handleChange = (next: string[]) => {
    const nextSet = new Set(next);
    // a previously-selected key that is gone from the new set was removed via its chip "x"
    for (const key of selectedKeys) {
      if (!nextSet.has(key)) {
        onRemove(key);
        return;
      }
    }
    // a key that appeared and is a known removed option was re-added from the dropdown
    for (const key of next) {
      if (!selectedKeys.includes(key) && removedKeys.includes(key)) {
        onReadd(key);
        return;
      }
    }
  };

  return (
    <>
      <SectionLabel>Selected resources</SectionLabel>
      <OverlayTipTop tip="Add files, entities, or repos from the list below to seed your dashboard with data. Then click create to view your dashboard.">
        <div>
          <SelectInputArray
            isCreatable={false}
            values={selectedKeys}
            options={removedKeys}
            valuesMap={valuesMap}
            onChange={handleChange}
            componentsOverride={removedKeys.length === 0 ? HIDDEN_MENU : undefined}
            defaultMessage="Browse and add resources below…"
          />
        </div>
      </OverlayTipTop>
    </>
  );
};

export default SelectionPanel;
