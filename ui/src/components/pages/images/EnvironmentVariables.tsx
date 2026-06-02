import React, { useMemo } from 'react';
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
import { OverlayTipRight } from '@components/shared/overlay/tips';

const TOOLTIP = `Environment variables that get mapped into the running image.`;

type EnvValue = Record<string, string | null>;

interface EnvironmentVariablesProps {
  value: EnvValue;
  onChange: (value: EnvValue) => void;
  mode: ImageFormMode;
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

const EnvironmentVariables: React.FC<EnvironmentVariablesProps> = ({ value, onChange, mode }) => {
  const entries = useMemo(
    () => (Object.keys(value).length ? Object.entries(value).map(([k, v]) => ({ key: k, value: v ?? '' })) : [{ key: '', value: '' }]),
    [value],
  );

  const handleUpdate = (newEntries: { key: string; value: string }[]) => {
    const result: EnvValue = {};
    for (const entry of newEntries) {
      if (entry.key) {
        result[entry.key] = entry.value === '' ? null : entry.value;
      }
    }
    onChange(result);
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
