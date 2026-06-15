import React from 'react';
import styled from 'styled-components';

// project imports
import {
  SectionRow,
  ValCol,
  EditSpacer,
  EditMiddle,
  EditFieldCol,
  TitleCol,
  FieldCol,
  FieldGroup,
  Label,
  ImageFieldsWrapper,
} from './shared.styled';
import { ImageFormMode } from './types';
import FieldBadge from '@components/shared/badges/FieldBadge';
import { OverlayTipRight } from '@components/shared/overlay/tips';

// spec: ./ImageInfo.spec.md

const TOOLTIPS = {
  self: `Path to the modifier folder for this image. Modifiers allow dynamic customization of image behavior at job execution time.`,
};

const KeyCol = styled.div`
  flex: 0 0 auto;
  min-width: 140px;
`;

const Input = styled.input`
  width: 100%;
  padding: 6px 12px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  font-size: 14px;

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

interface ModifiersProps {
  value: string;
  onChange: (value: string) => void;
  mode: ImageFormMode;
}

const ModifierFields: React.FC<{
  value: string;
  onChange: (value: string) => void;
}> = ({ value, onChange }) => (
  <ImageFieldsWrapper>
    <FieldGroup>
      <Label>Modifiers Path</Label>
      <OverlayTipRight tip={TOOLTIPS.self}>
        <Input type="text" value={value ?? ''} placeholder="/path/to/modifiers" onChange={(e) => onChange(e.target.value)} />
      </OverlayTipRight>
    </FieldGroup>
  </ImageFieldsWrapper>
);

const Modifiers: React.FC<ModifiersProps> = ({ value, onChange, mode }) => {
  if (mode === ImageFormMode.View) {
    return (
      <SectionRow>
        <KeyCol>
          <OverlayTipRight tip={TOOLTIPS.self}>
            <b>Modifiers</b>
          </OverlayTipRight>
        </KeyCol>
        <ValCol>
          <FieldBadge field={value || 'None'} color="#7e7c7c" />
        </ValCol>
      </SectionRow>
    );
  }

  if (mode === ImageFormMode.Edit) {
    return (
      <SectionRow>
        <EditSpacer />
        <EditMiddle>
          <OverlayTipRight tip={TOOLTIPS.self}>
            <b>Modifiers</b>
          </OverlayTipRight>
        </EditMiddle>
        <EditFieldCol>
          <ModifierFields value={value} onChange={onChange} />
        </EditFieldCol>
      </SectionRow>
    );
  }

  return (
    <SectionRow>
      <TitleCol>
        <h5>Modifiers</h5>
      </TitleCol>
      <FieldCol>
        <ModifierFields value={value} onChange={onChange} />
      </FieldCol>
    </SectionRow>
  );
};

export default Modifiers;
