import React from 'react';
import styled from 'styled-components';

// project imports
import { SectionRow, ValCol, EditSpacer, EditMiddle, EditFieldCol, TitleCol, FieldCol, ImageFieldsWrapper } from './shared.styled';
import { ImageFormMode } from './types';
import FieldBadge from '@components/shared/badges/FieldBadge';
import SelectableArray from '@components/shared/inputs/selectable/SelectableArray';
import { OverlayTipRight } from '@components/shared/overlay/tips';

const TOOLTIP = `Network policies dictate which entities the image can connect to or receive communication from.`;

const KeyCol = styled.div`
  flex: 0 0 auto;
  min-width: 140px;
`;

interface NetworkPoliciesProps {
  value: string[];
  onChange: (value: string[]) => void;
  mode: ImageFormMode;
  disabled?: boolean;
}

const DisplayNetworkPolicies: React.FC<{ policies: string[] }> = ({ policies }) => (
  <SectionRow>
    <KeyCol>
      <OverlayTipRight tip={TOOLTIP}>
        <b>Network Policies</b>
      </OverlayTipRight>
    </KeyCol>
    <ValCol>
      <FieldBadge field={policies.length ? policies : 'None'} color="#7e7c7c" />
    </ValCol>
  </SectionRow>
);

const NetworkPolicies: React.FC<NetworkPoliciesProps> = ({ value, onChange, mode, disabled }) => {
  if (mode === ImageFormMode.View) {
    return <DisplayNetworkPolicies policies={value} />;
  }

  const isEdit = mode === ImageFormMode.Edit;

  if (isEdit) {
    return (
      <SectionRow>
        <EditSpacer />
        <EditMiddle>
          <OverlayTipRight tip={TOOLTIP}>
            <b>Network Policies</b>
          </OverlayTipRight>
        </EditMiddle>
        <EditFieldCol>
          <ImageFieldsWrapper>
            <SelectableArray
              initialEntries={value}
              setEntries={onChange}
              disabled={disabled ?? false}
              placeholder="example-policy"
              trim={true}
            />
          </ImageFieldsWrapper>
        </EditFieldCol>
      </SectionRow>
    );
  }

  return (
    <SectionRow>
      <TitleCol>
        <h5>Network Policies</h5>
      </TitleCol>
      <FieldCol>
        <ImageFieldsWrapper>
          <OverlayTipRight tip={TOOLTIP}>
            <SelectableArray
              initialEntries={value}
              setEntries={onChange}
              disabled={disabled ?? false}
              placeholder="example-policy"
              trim={true}
            />
          </OverlayTipRight>
        </ImageFieldsWrapper>
      </FieldCol>
    </SectionRow>
  );
};

export default NetworkPolicies;
