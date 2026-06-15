import React, { useState } from 'react';
import styled from 'styled-components';

// project imports
import {
  SectionRow,
  IndentCol,
  ValCol,
  EditSpacer,
  EditMiddle,
  EditFieldCol,
  TitleCol,
  FieldCol,
  ImageFieldsWrapper,
} from './shared.styled';
import { ImageFormMode } from './types';
import FieldBadge from '@components/shared/badges/FieldBadge';
import SelectableDictionary from '@components/shared/inputs/selectable/SelectableDictionary';
import type { DictionaryEntry } from '@components/shared/inputs/selectable/SelectableDictionary';
import { OverlayTipRight } from '@components/shared/overlay/tips';

// spec: ./ImageInfo.spec.md

const TOOLTIP = `Environment variables that get mapped into the running image.`;

type EnvValue = Record<string, string | null>;

interface EnvironmentVariablesProps {
  value: EnvValue;
  onChange: (value: EnvValue) => void;
  mode: ImageFormMode;
  resetKey?: number;
}

/**
 * Convert an env dictionary into editable key/value rows, always seeding a single empty row when
 * the dictionary has no entries so the form starts with an editable line.
 *
 * @param value - The env dictionary (values may be `null` for empty values).
 * @returns The key/value rows for `SelectableDictionary`.
 */
export function valueToEntries(value: EnvValue): DictionaryEntry[] {
  const keys = Object.keys(value);
  if (keys.length === 0) {
    return [{ key: '', value: '' }];
  }
  return keys.map((k) => ({ key: k, value: value[k] ?? '' }));
}

/**
 * Convert editable key/value rows back into an env dictionary, dropping rows with an empty key
 * (in-progress trailing rows) and storing empty values as `null` to match the API shape.
 *
 * @param entries - The key/value rows from `SelectableDictionary`.
 * @returns The env dictionary suitable for the image update/create payload.
 */
export function entriesToDict(entries: DictionaryEntry[]): EnvValue {
  const result: EnvValue = {};
  for (const entry of entries) {
    if (entry.key) {
      result[entry.key] = entry.value === '' ? null : entry.value;
    }
  }
  return result;
}

const KeyCol = styled.div`
  flex: 0 0 auto;
  min-width: 140px;
`;

const EnvKeyCol = styled.div`
  flex: 0 0 auto;
  min-width: 120px;
`;

const DisplayEnvironmentVars: React.FC<{ value: EnvValue }> = ({ value }) => {
  const entries = Object.entries(value);
  if (entries.length === 0) {
    return (
      <SectionRow>
        <KeyCol>
          <OverlayTipRight tip={TOOLTIP}>
            <b>Environment</b>
          </OverlayTipRight>
        </KeyCol>
        <ValCol>
          <FieldBadge field="None" color="#7e7c7c" />
        </ValCol>
      </SectionRow>
    );
  }
  return (
    <>
      <SectionRow>
        <div>
          <OverlayTipRight tip={TOOLTIP}>
            <b>Environment</b>
          </OverlayTipRight>
        </div>
      </SectionRow>
      {entries.map(([key, val]) => (
        <SectionRow key={key}>
          <IndentCol />
          <EnvKeyCol>
            <em>{`${key}: `}</em>
          </EnvKeyCol>
          <ValCol>
            <FieldBadge field={val} color="#7e7c7c" />
          </ValCol>
        </SectionRow>
      ))}
    </>
  );
};

const EnvironmentVariables: React.FC<EnvironmentVariablesProps> = ({ value, onChange, mode, resetKey }) => {
  // Rows are held in local state (not re-derived from `value` each render) so the trailing empty
  // row that SelectableDictionary appends survives — the env dictionary can't represent an
  // empty-key row, so deriving from it would immediately drop the new row and break auto-add.
  const [entries, setEntries] = useState<DictionaryEntry[]>(() => valueToEntries(value));
  // Re-seed the rows when the parent signals a fresh dataset (e.g. after a save refetch),
  // without clobbering in-progress edits.
  const [prevResetKey, setPrevResetKey] = useState(resetKey);
  if (resetKey !== prevResetKey) {
    setPrevResetKey(resetKey);
    setEntries(valueToEntries(value));
  }

  const handleUpdate = (newEntries: DictionaryEntry[]) => {
    setEntries(newEntries);
    onChange(entriesToDict(newEntries));
  };

  if (mode === ImageFormMode.View) {
    return <DisplayEnvironmentVars value={value} />;
  }

  const isEdit = mode === ImageFormMode.Edit;

  if (isEdit) {
    return (
      <SectionRow>
        <EditSpacer />
        <EditMiddle>
          <OverlayTipRight tip={TOOLTIP}>
            <b>Environment</b>
          </OverlayTipRight>
        </EditMiddle>
        <EditFieldCol>
          <ImageFieldsWrapper>
            <SelectableDictionary
              entries={entries}
              disabled={false}
              setEntries={handleUpdate}
              keyPlaceholder="New Variable"
              valuePlaceholder="New Value"
              trim={true}
              keys={undefined}
              deleted={undefined}
              setDeleted={undefined}
            />
          </ImageFieldsWrapper>
        </EditFieldCol>
      </SectionRow>
    );
  }

  return (
    <SectionRow>
      <TitleCol>
        <h5>Environment</h5>
      </TitleCol>
      <FieldCol>
        <ImageFieldsWrapper>
          <OverlayTipRight tip={TOOLTIP}>
            <SelectableDictionary
              entries={entries}
              disabled={false}
              setEntries={handleUpdate}
              keyPlaceholder="New Variable"
              valuePlaceholder="New Value"
              trim={true}
              keys={undefined}
              deleted={undefined}
              setDeleted={undefined}
            />
          </OverlayTipRight>
        </ImageFieldsWrapper>
      </FieldCol>
    </SectionRow>
  );
};

export default EnvironmentVariables;
