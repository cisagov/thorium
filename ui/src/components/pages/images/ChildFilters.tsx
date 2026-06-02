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
  FieldGroup,
  Label,
  SwitchRow,
  ImageFieldsWrapper,
} from './shared.styled';
import { ImageFormMode } from './types';
import FieldBadge from '@components/shared/badges/FieldBadge';
import ToggleSwitch from '@components/shared/inputs/ToggleSwitch';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import type { ChildFilters as ChildFiltersType } from '@models/images';

const TOOLTIPS = {
  self: `Regex filters for children produced by this image. If no filters are given, all children will be uploaded. Regular expressions must conform to Rust regex crate standards.`,
  mime: `Regex filters applied to the MIME type of child files (e.g., 'application/pdf', 'image/.*').`,
  file_name: `Regex filters applied to the full file name including extension (e.g., '.*\\.exe', 'report_.*').`,
  file_extension: `Regex filters applied to the file extension without the leading dot (e.g., 'txt', 'dll', 'so').`,
  submit_non_matches: `When enabled, children that do NOT match any filter will be submitted instead of those that do.`,
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

const SwitchLabel = styled.span`
  font-size: 13px;
  font-weight: 600;
  color: var(--thorium-secondary-text);
  min-width: 220px;
`;

const BadgeWrap = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
`;

interface ChildFiltersProps {
  value: ChildFiltersType;
  onChange: (value: ChildFiltersType) => void;
  onValidate?: (hasErrors: boolean) => void;
  mode: ImageFormMode;
  disabled?: boolean;
  resetKey?: number;
}

function toArray(val: unknown): string[] {
  if (Array.isArray(val)) return val as string[];
  return [];
}

const DisplayChildFilters: React.FC<{ value: ChildFiltersType }> = ({ value }) => (
  <>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <OverlayTipRight tip={TOOLTIPS.mime}>
          <em>mime: </em>
        </OverlayTipRight>
      </KeyCol>
      <ValCol>
        <BadgeWrap>
          {toArray(value.mime).length > 0 ? (
            toArray(value.mime).map((m) => <FieldBadge key={m} field={m} color="#7e7c7c" />)
          ) : (
            <FieldBadge field="None" color="#7e7c7c" />
          )}
        </BadgeWrap>
      </ValCol>
    </SectionRow>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <OverlayTipRight tip={TOOLTIPS.file_name}>
          <em>file_name: </em>
        </OverlayTipRight>
      </KeyCol>
      <ValCol>
        <BadgeWrap>
          {toArray(value.file_name).length > 0 ? (
            toArray(value.file_name).map((f) => <FieldBadge key={f} field={f} color="#7e7c7c" />)
          ) : (
            <FieldBadge field="None" color="#7e7c7c" />
          )}
        </BadgeWrap>
      </ValCol>
    </SectionRow>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <OverlayTipRight tip={TOOLTIPS.file_extension}>
          <em>file_extension: </em>
        </OverlayTipRight>
      </KeyCol>
      <ValCol>
        <BadgeWrap>
          {toArray(value.file_extension).length > 0 ? (
            toArray(value.file_extension).map((f) => <FieldBadge key={f} field={f} color="#7e7c7c" />)
          ) : (
            <FieldBadge field="None" color="#7e7c7c" />
          )}
        </BadgeWrap>
      </ValCol>
    </SectionRow>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <OverlayTipRight tip={TOOLTIPS.submit_non_matches}>
          <em>submit_non_matches: </em>
        </OverlayTipRight>
      </KeyCol>
      <ValCol>
        <FieldBadge field={value.submit_non_matches ?? false} color="#7e7c7c" />
      </ValCol>
    </SectionRow>
  </>
);

function parseCommaSeparated(text: string): string[] {
  return text
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

const ChildFiltersFields: React.FC<{
  value: ChildFiltersType;
  onChange: (value: ChildFiltersType) => void;
  disabled: boolean;
  resetKey?: number;
}> = ({ value, onChange, disabled, resetKey }) => {
  const [mimeText, setMimeText] = useState(() => toArray(value.mime).join(', '));
  const [fileNameText, setFileNameText] = useState(() => toArray(value.file_name).join(', '));
  const [fileExtText, setFileExtText] = useState(() => toArray(value.file_extension).join(', '));
  // Re-derive the text fields from value when the parent signals a fresh dataset
  // (e.g. after a save refetch), without clobbering in-progress edits.
  const [prevResetKey, setPrevResetKey] = useState(resetKey);
  if (resetKey !== prevResetKey) {
    setPrevResetKey(resetKey);
    setMimeText(toArray(value.mime).join(', '));
    setFileNameText(toArray(value.file_name).join(', '));
    setFileExtText(toArray(value.file_extension).join(', '));
  }

  const handleMime = (text: string) => {
    setMimeText(text);
    onChange({ ...value, mime: parseCommaSeparated(text) });
  };

  const handleFileName = (text: string) => {
    setFileNameText(text);
    onChange({ ...value, file_name: parseCommaSeparated(text) });
  };

  const handleFileExt = (text: string) => {
    setFileExtText(text);
    onChange({ ...value, file_extension: parseCommaSeparated(text) });
  };

  return (
    <ImageFieldsWrapper>
      <FieldGroup>
        <Label>MIME Filters</Label>
        <OverlayTipRight tip={TOOLTIPS.mime}>
          <Input
            type="text"
            value={mimeText}
            placeholder="application/pdf, image/.*"
            disabled={disabled}
            onChange={(e) => handleMime(e.target.value)}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>File Name Filters</Label>
        <OverlayTipRight tip={TOOLTIPS.file_name}>
          <Input
            type="text"
            value={fileNameText}
            placeholder=".*\.exe, report_.*"
            disabled={disabled}
            onChange={(e) => handleFileName(e.target.value)}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>File Extension Filters</Label>
        <OverlayTipRight tip={TOOLTIPS.file_extension}>
          <Input
            type="text"
            value={fileExtText}
            placeholder="txt, dll, so"
            disabled={disabled}
            onChange={(e) => handleFileExt(e.target.value)}
          />
        </OverlayTipRight>
      </FieldGroup>
      <SwitchRow>
        <SwitchLabel>Submit Non-Matches</SwitchLabel>
        <OverlayTipRight tip={TOOLTIPS.submit_non_matches}>
          <ToggleSwitch
            checked={value.submit_non_matches ?? false}
            disabled={disabled}
            onChange={() => onChange({ ...value, submit_non_matches: !value.submit_non_matches })}
          />
        </OverlayTipRight>
      </SwitchRow>
    </ImageFieldsWrapper>
  );
};

const ChildFilters: React.FC<ChildFiltersProps> = ({ value, onChange, mode, disabled = false, resetKey }) => {
  const isEmpty =
    !value ||
    (toArray(value.mime).length === 0 &&
      toArray(value.file_name).length === 0 &&
      toArray(value.file_extension).length === 0 &&
      !value.submit_non_matches);

  if (mode === ImageFormMode.View) {
    return (
      <>
        <SectionRow>
          <KeyCol>
            <OverlayTipRight tip={TOOLTIPS.self}>
              <b>Child Filters</b>
            </OverlayTipRight>
          </KeyCol>
          <ValCol>{isEmpty ? <FieldBadge field="None" color="#7e7c7c" /> : null}</ValCol>
        </SectionRow>
        {!isEmpty && <DisplayChildFilters value={value} />}
      </>
    );
  }

  if (mode === ImageFormMode.Edit) {
    return (
      <SectionRow>
        <EditSpacer />
        <EditMiddle>
          <OverlayTipRight tip={TOOLTIPS.self}>
            <b>Child Filters</b>
          </OverlayTipRight>
        </EditMiddle>
        <EditFieldCol>
          <ChildFiltersFields value={value ?? {}} onChange={onChange} disabled={disabled} resetKey={resetKey} />
        </EditFieldCol>
      </SectionRow>
    );
  }

  return (
    <SectionRow>
      <TitleCol>
        <h5>Child Filters</h5>
      </TitleCol>
      <FieldCol>
        <ChildFiltersFields value={value ?? {}} onChange={onChange} disabled={disabled} />
      </FieldCol>
    </SectionRow>
  );
};

export default ChildFilters;
