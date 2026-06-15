import React from 'react';
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
  FieldGroup,
  Label,
  SwitchRow,
  ImageFieldsWrapper,
} from './shared.styled';
import { ImageFormMode } from './types';
import FieldBadge from '@components/shared/badges/FieldBadge';
import ToggleSwitch from '@components/shared/inputs/ToggleSwitch';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import type { SecurityContext as SecurityContextType } from '@models/images';

// spec: ./ImageInfo.spec.md

const TOOLTIPS = {
  self: `Runtime security context settings. Only admins can adjust these image settings.`,
  user: `The user ID used to run this image. This is an admin only feature.`,
  group: `The group ID used to run this image. This is an admin only feature.`,
  allow_privilege_escalation: `Whether privilege escalation is allowed when executing the image. Default: false.`,
};

const KeyCol = styled.div`
  flex: 0 0 auto;
  min-width: 120px;
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

const SwitchLabel = styled.span`
  font-size: 13px;
  font-weight: 600;
  color: var(--thorium-secondary-text);
  min-width: 220px;
`;

interface SecurityContextProps {
  value: SecurityContextType;
  onChange: (value: SecurityContextType) => void;
  mode: ImageFormMode;
  disabled?: boolean;
}

const DisplaySecurityContext: React.FC<{ value: SecurityContextType }> = ({ value }) => (
  <>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <em>user: </em>
      </KeyCol>
      <ValCol>
        <FieldBadge field={value.user} color="DarkRed" />
      </ValCol>
    </SectionRow>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <em>group: </em>
      </KeyCol>
      <ValCol>
        <FieldBadge field={value.group} color="DarkRed" />
      </ValCol>
    </SectionRow>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <em>allow_privilege_escalation: </em>
      </KeyCol>
      <ValCol>
        <FieldBadge field={value.allow_privilege_escalation} color="DarkRed" />
      </ValCol>
    </SectionRow>
  </>
);

const SecurityContextFields: React.FC<{
  value: SecurityContextType;
  onChange: (value: SecurityContextType) => void;
  disabled: boolean;
}> = ({ value, onChange, disabled }) => {
  const update = (key: keyof SecurityContextType, val: SecurityContextType[keyof SecurityContextType]) => {
    onChange({ ...value, [key]: val });
  };

  return (
    <ImageFieldsWrapper>
      <FieldGroup>
        <Label>Run As User</Label>
        <OverlayTipRight tip={TOOLTIPS.user}>
          <Input
            type="text"
            value={value.user != null ? String(value.user) : ''}
            placeholder="99999"
            disabled={disabled}
            onChange={(e) => {
              const cleaned = e.target.value.replace(/[^0-9]/g, '');
              update('user', cleaned === '' ? undefined : Number(cleaned));
            }}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>Run As Group</Label>
        <OverlayTipRight tip={TOOLTIPS.group}>
          <Input
            type="text"
            value={value.group != null ? String(value.group) : ''}
            placeholder="99999"
            disabled={disabled}
            onChange={(e) => {
              const cleaned = e.target.value.replace(/[^0-9]/g, '');
              update('group', cleaned === '' ? undefined : Number(cleaned));
            }}
          />
        </OverlayTipRight>
      </FieldGroup>
      <SwitchRow>
        <SwitchLabel>Allow Privilege Escalation</SwitchLabel>
        <OverlayTipRight tip={TOOLTIPS.allow_privilege_escalation}>
          <ToggleSwitch
            checked={value.allow_privilege_escalation ?? false}
            disabled={disabled}
            onChange={() => update('allow_privilege_escalation', !value.allow_privilege_escalation)}
          />
        </OverlayTipRight>
      </SwitchRow>
    </ImageFieldsWrapper>
  );
};

const SecurityContext: React.FC<SecurityContextProps> = ({ value, onChange, mode, disabled = false }) => {
  if (mode === ImageFormMode.View) {
    return (
      <>
        <SectionRow>
          <div>
            <OverlayTipRight tip={TOOLTIPS.self}>
              <b>Security Context</b>
            </OverlayTipRight>
          </div>
        </SectionRow>
        <DisplaySecurityContext value={value} />
      </>
    );
  }

  if (mode === ImageFormMode.Edit) {
    return (
      <SectionRow>
        <EditSpacer />
        <EditMiddle>
          <OverlayTipRight tip={TOOLTIPS.self}>
            <b>Security Context</b>
          </OverlayTipRight>
        </EditMiddle>
        <EditFieldCol>
          <SecurityContextFields value={value} onChange={onChange} disabled={disabled} />
        </EditFieldCol>
      </SectionRow>
    );
  }

  return (
    <SectionRow>
      <TitleCol>
        <h5>Security Context</h5>
      </TitleCol>
      <FieldCol>
        <SecurityContextFields value={value} onChange={onChange} disabled={disabled} />
      </FieldCol>
    </SectionRow>
  );
};

export default SecurityContext;
