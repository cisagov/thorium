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
  ImageFieldsWrapper,
} from './shared.styled';
import { ImageFormMode } from './types';
import FieldBadge from '@components/shared/badges/FieldBadge';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import type { Kvm } from '@models/images';

const TOOLTIPS = {
  self: `KVM virtual machine configuration for images using the Kvm scaler.`,
  xml: `Path to the golden libvirt XML file defining the VM configuration.`,
  qcow2: `Path to the golden qcow2 disk image to use as the VM backing store.`,
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

interface KVMProps {
  value: Kvm;
  onChange: (value: Kvm) => void;
  mode: ImageFormMode;
}

const DisplayKVM: React.FC<{ value: Kvm }> = ({ value }) => (
  <>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <OverlayTipRight tip={TOOLTIPS.xml}>
          <em>xml: </em>
        </OverlayTipRight>
      </KeyCol>
      <ValCol>
        <FieldBadge field={value.xml || 'None'} color="#7e7c7c" />
      </ValCol>
    </SectionRow>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <OverlayTipRight tip={TOOLTIPS.qcow2}>
          <em>qcow2: </em>
        </OverlayTipRight>
      </KeyCol>
      <ValCol>
        <FieldBadge field={value.qcow2 || 'None'} color="#7e7c7c" />
      </ValCol>
    </SectionRow>
  </>
);

const KVMFields: React.FC<{
  value: Kvm;
  onChange: (value: Kvm) => void;
}> = ({ value, onChange }) => {
  const update = (key: keyof Kvm, val: string) => {
    onChange({ ...value, [key]: val });
  };

  return (
    <ImageFieldsWrapper>
      <FieldGroup>
        <Label>XML Path</Label>
        <OverlayTipRight tip={TOOLTIPS.xml}>
          <Input type="text" value={value.xml ?? ''} placeholder="/path/to/vm.xml" onChange={(e) => update('xml', e.target.value)} />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>QCOW2 Path</Label>
        <OverlayTipRight tip={TOOLTIPS.qcow2}>
          <Input
            type="text"
            value={value.qcow2 ?? ''}
            placeholder="/path/to/disk.qcow2"
            onChange={(e) => update('qcow2', e.target.value)}
          />
        </OverlayTipRight>
      </FieldGroup>
    </ImageFieldsWrapper>
  );
};

const KVM: React.FC<KVMProps> = ({ value, onChange, mode }) => {
  if (mode === ImageFormMode.View) {
    return (
      <>
        <SectionRow>
          <div>
            <OverlayTipRight tip={TOOLTIPS.self}>
              <b>KVM</b>
            </OverlayTipRight>
          </div>
        </SectionRow>
        <DisplayKVM value={value} />
      </>
    );
  }

  if (mode === ImageFormMode.Edit) {
    return (
      <SectionRow>
        <EditSpacer />
        <EditMiddle>
          <OverlayTipRight tip={TOOLTIPS.self}>
            <b>KVM</b>
          </OverlayTipRight>
        </EditMiddle>
        <EditFieldCol>
          <KVMFields value={value ?? { xml: '', qcow2: '' }} onChange={onChange} />
        </EditFieldCol>
      </SectionRow>
    );
  }

  return (
    <SectionRow>
      <TitleCol>
        <h5>KVM</h5>
      </TitleCol>
      <FieldCol>
        <KVMFields value={value ?? { xml: '', qcow2: '' }} onChange={onChange} />
      </FieldCol>
    </SectionRow>
  );
};

export default KVM;
