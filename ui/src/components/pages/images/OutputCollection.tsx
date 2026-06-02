import React, { useState, useEffect, useMemo } from 'react';
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
  Label,
  ImageFieldsWrapper,
} from './shared.styled';
import { ImageFormMode } from './types';
import AlertBanner from '@components/shared/alerts/AlertBanner';
import FieldBadge from '@components/shared/badges/FieldBadge';
import SelectableArray from '@components/shared/inputs/selectable/SelectableArray';
import ToggleSwitch from '@components/shared/inputs/ToggleSwitch';
import SelectGroups from '@components/pages/groups/SelectGroups';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import { AUTO_TAG_LOGIC_VALUES } from '@utilities/rules/tools/image/schema';
import type { AutoTag } from '@models/images';
import { OutputHandler } from '@models/images';
import type { OutputCollection as OutputCollectionType } from '@models/results';

const TOOLTIPS = {
  self: `Configurations that determine how the Thorium agent will intake analysis artifacts after running this image.`,
  files: {
    names: `Names of specific result files to collect. The default behavior collects all files in the result files directory.`,
    tags: `Path to a JSON formatted file of key/value tags.`,
    results: `Path to a single renderable results file (JSON, Table, String, etc).`,
    result_files: `Path to a directory containing one or more result file(s). These are displayed as download links.`,
  },
  groups: `Limit this image to uploading results to these selected groups. By default results are uploaded to the group(s) of the file or repo that the image ran on.`,
  children: `Path to children samples extracted by the image.`,
  auto_tag: `Specific keys that will get automatically added as tags from the image results. JSON formatted results are required to use auto tagging.`,
  as_filesystem: `Preserve full directory structure by uploading children files and folders as a filesystem entity.`,
};

const KeyCol = styled.div`
  flex: 0 0 auto;
  min-width: 120px;
`;

const SubIndent = styled.div`
  flex: 0 0 auto;
  min-width: 40px;
`;

const SubKeyCol = styled.div`
  flex: 0 0 auto;
  min-width: 100px;
`;

const SubValCol = styled.div`
  flex: 1;
`;

const FieldGroup = styled.div`
  margin-top: 4px;
`;

const AutoTagRow = styled.div`
  display: flex;
  gap: 6px;
  align-items: center;
  margin-bottom: 4px;
`;

const Select = styled.select`
  padding: 6px 8px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  font-size: 13px;
  min-width: 110px;

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const RemoveBtn = styled.button`
  background: none;
  border: none;
  color: var(--thorium-secondary-text);
  cursor: pointer;
  font-size: 1.1rem;
  padding: 0 4px;
  line-height: 1;

  &:hover {
    color: var(--thorium-danger, #e74c3c);
  }
`;

const AddBtn = styled.button`
  background: none;
  border: 1px dashed var(--thorium-panel-border);
  border-radius: 4px;
  color: var(--thorium-secondary-text);
  cursor: pointer;
  font-size: 13px;
  padding: 4px 12px;
  margin-top: 4px;

  &:hover {
    color: var(--thorium-text);
    border-color: var(--thorium-highlight-panel-border);
  }
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

const SubtitleRow = styled.div`
  display: flex;
  margin-top: 8px;
  margin-bottom: 8px;
`;

interface AutoTagEntry {
  key: string;
  value: string;
  logic: string;
}

interface FormOutputCollection {
  files: {
    results: string;
    result_files: string;
    names: string[];
    tags: string;
  };
  children: string;
  as_filesystem: boolean;
  select_auto_tag: AutoTagEntry[];
  select_groups: Record<string, boolean>;
}

interface OutputCollectionProps {
  value: OutputCollectionType;
  onChange: (value: OutputCollectionType) => void;
  onValidate?: (hasErrors: boolean) => void;
  groups: string[];
  mode: ImageFormMode;
  disabled?: boolean;
  resetKey?: number;
}

function apiToForm(oc: OutputCollectionType, availableGroups: string[]): FormOutputCollection {
  const selectedGroups: Record<string, boolean> = {};
  for (const g of availableGroups) {
    selectedGroups[g] = false;
  }
  if (oc.groups) {
    for (const g of oc.groups) {
      selectedGroups[g] = true;
    }
  }

  const autoTagEntries: AutoTagEntry[] = [];
  if (oc.auto_tag) {
    for (const [key, tag] of Object.entries(oc.auto_tag)) {
      autoTagEntries.push({ key, value: tag.key ?? '', logic: typeof tag.logic === 'string' ? tag.logic : 'Exists' });
    }
  }

  return {
    files: {
      results: oc.files?.results ?? '',
      result_files: oc.files?.result_files ?? '',
      names: oc.files?.names ?? [],
      tags: oc.files?.tags ?? '',
    },
    children: oc.children ?? '',
    as_filesystem: oc.as_filesystem ?? false,
    select_auto_tag: autoTagEntries,
    select_groups: selectedGroups,
  };
}

function formToApi(form: FormOutputCollection): OutputCollectionType {
  // Seed all API-required fields with their defaults; the blocks below override with form values
  // and the empty defaults are stripped/replaced as needed.
  const result: OutputCollectionType = {
    handler: OutputHandler.Files,
    files: {},
    as_filesystem: form.as_filesystem,
    children: '',
    auto_tag: {},
    groups: [],
  };

  const files: Partial<NonNullable<OutputCollectionType['files']>> = {};
  if (form.files.results) files.results = form.files.results;
  if (form.files.result_files) files.result_files = form.files.result_files;
  const filteredNames = (form.files.names ?? []).filter((n) => n !== '');
  if (filteredNames.length > 0) files.names = filteredNames;
  if (form.files.tags) files.tags = form.files.tags;
  if (Object.keys(files).length > 0) result.files = files;

  if (form.children) result.children = form.children;

  const autoTag: Record<string, AutoTag> = {};
  for (const entry of form.select_auto_tag) {
    if (entry.key.trim()) {
      autoTag[entry.key] = {
        logic: (entry.logic || 'Exists') as AutoTag['logic'],
        key: entry.value.trim() || undefined,
      };
    }
  }
  if (Object.keys(autoTag).length > 0) result.auto_tag = autoTag;

  const groups = Object.entries(form.select_groups)
    .filter(([, selected]) => selected)
    .map(([name]) => name);
  if (groups.length > 0) result.groups = groups;

  return result;
}

function validateOC(form: FormOutputCollection): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const tag of form.select_auto_tag) {
    if (tag.key.trim() === '' && tag.value.trim().length > 0) {
      errors.auto_tag = 'Tag Names Can Not Be Empty';
      break;
    }
  }
  return errors;
}

const DisplayOutputCollection: React.FC<{ value: OutputCollectionType }> = ({ value }) => {
  if (!value || Object.keys(value).length === 0) return null;

  return (
    <>
      <SectionRow>
        <IndentCol />
        <KeyCol>
          <em>files</em>
        </KeyCol>
        <ValCol />
      </SectionRow>
      <SectionRow>
        <SubIndent />
        <SubKeyCol>
          <em>results</em>
        </SubKeyCol>
        <SubValCol>
          <FieldBadge field={value.files?.results} color="#7e7c7c" />
        </SubValCol>
      </SectionRow>
      <SectionRow>
        <SubIndent />
        <SubKeyCol>
          <em>result_files</em>
        </SubKeyCol>
        <SubValCol>
          <FieldBadge field={value.files?.result_files} color="#7e7c7c" />
        </SubValCol>
      </SectionRow>
      {value.files?.names && value.files.names.length > 0 && value.files.names[0] !== '' && (
        <SectionRow>
          <SubIndent />
          <SubKeyCol>
            <em>file_names</em>
          </SubKeyCol>
          <SubValCol>
            <FieldBadge field={value.files.names} color="#7e7c7c" />
          </SubValCol>
        </SectionRow>
      )}
      <SectionRow>
        <SubIndent />
        <SubKeyCol>
          <em>tags</em>
        </SubKeyCol>
        <SubValCol>
          <FieldBadge field={value.files?.tags} color="#7e7c7c" />
        </SubValCol>
      </SectionRow>
      <SectionRow>
        <IndentCol />
        <KeyCol>
          <em>children</em>
        </KeyCol>
        <ValCol>
          <FieldBadge field={value.children} color="#7e7c7c" />
        </ValCol>
      </SectionRow>
      <SectionRow>
        <IndentCol />
        <KeyCol>
          <em>as filesystem</em>
        </KeyCol>
        <ValCol>
          <FieldBadge field={value.as_filesystem} color="#7e7c7c" />
        </ValCol>
      </SectionRow>
      {value.auto_tag && Object.keys(value.auto_tag).length > 0 && (
        <SectionRow>
          <IndentCol />
          <KeyCol>
            <em>auto tagging: </em>
          </KeyCol>
          <ValCol>
            {Object.entries(value.auto_tag).map(([key, tag]) => {
              const logic = typeof tag.logic === 'string' ? tag.logic : 'Exists';
              return (
                <span
                  key={key}
                  style={{
                    backgroundColor: '#7e7c7c',
                    color: 'white',
                    padding: '2px 8px',
                    borderRadius: '4px',
                    marginRight: '4px',
                    fontSize: '12px',
                  }}
                >
                  {`${key} (${logic})${tag.key ? `: ${tag.key}` : ''}`}
                </span>
              );
            })}
          </ValCol>
        </SectionRow>
      )}
      {value.groups && value.groups.length > 0 && (
        <SectionRow>
          <IndentCol />
          <KeyCol>
            <em>groups: </em>
          </KeyCol>
          <ValCol>
            <FieldBadge field={value.groups} color="#7e7c7c" />
          </ValCol>
        </SectionRow>
      )}
    </>
  );
};

const OutputCollectionInputs: React.FC<{
  form: FormOutputCollection;
  setForm: (f: FormOutputCollection) => void;
  errors: Record<string, string>;
  disabled: boolean;
}> = ({ form, setForm, errors, disabled }) => {
  const update = (key: string, subkey: string, val: unknown) => {
    const copy = structuredClone(form);
    if (subkey) {
      (copy as unknown as Record<string, Record<string, unknown>>)[key][subkey] = val;
    } else {
      (copy as unknown as Record<string, unknown>)[key] = val;
    }
    setForm(copy);
  };

  return (
    <ImageFieldsWrapper>
      <FieldGroup>
        <Label>Results</Label>
        <OverlayTipRight tip={TOOLTIPS.files.results}>
          <Input
            type="text"
            value={form.files.results}
            placeholder="/tmp/thorium/results"
            disabled={disabled}
            onChange={(e) => update('files', 'results', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>Result Files</Label>
        <OverlayTipRight tip={TOOLTIPS.files.result_files}>
          <Input
            type="text"
            value={form.files.result_files}
            placeholder="/tmp/thorium/result-files"
            disabled={disabled}
            onChange={(e) => update('files', 'result_files', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>Result File Names</Label>
        <OverlayTipRight tip={TOOLTIPS.files.names}>
          <SelectableArray
            initialEntries={form.files.names}
            setEntries={(names: string[]) => update('files', 'names', names)}
            disabled={disabled}
            placeholder="file name"
            trim={false}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>Children</Label>
        <OverlayTipRight tip={TOOLTIPS.children}>
          <Input
            type="text"
            value={form.children}
            placeholder="/tmp/thorium/children"
            disabled={disabled}
            onChange={(e) => update('children', '', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>As Filesystem</Label>
        <OverlayTipRight tip={TOOLTIPS.as_filesystem}>
          <ToggleSwitch checked={form.as_filesystem} onChange={() => update('as_filesystem', '', !form.as_filesystem)} />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>Tags</Label>
        <OverlayTipRight tip={TOOLTIPS.files.tags}>
          <Input
            type="text"
            value={form.files.tags}
            placeholder="/tmp/thorium/tags"
            disabled={disabled}
            onChange={(e) => update('files', 'tags', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>
      <SubtitleRow>
        <Label>Auto Tagging</Label>
      </SubtitleRow>
      <OverlayTipRight tip={TOOLTIPS.auto_tag}>
        <div>
          {form.select_auto_tag.map((entry, idx) => (
            <AutoTagRow key={idx}>
              <Input
                style={{ flex: 1 }}
                placeholder="tag name"
                value={entry.key}
                disabled={disabled}
                onChange={(e) => {
                  const copy = [...form.select_auto_tag];
                  copy[idx] = { ...copy[idx], key: e.target.value };
                  update('select_auto_tag', '', copy);
                }}
              />
              <Select
                value={entry.logic}
                disabled={disabled}
                onChange={(e) => {
                  const copy = [...form.select_auto_tag];
                  copy[idx] = { ...copy[idx], logic: e.target.value };
                  update('select_auto_tag', '', copy);
                }}
              >
                {AUTO_TAG_LOGIC_VALUES.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </Select>
              <Input
                style={{ flex: 1 }}
                placeholder="key (optional)"
                value={entry.value}
                disabled={disabled}
                onChange={(e) => {
                  const copy = [...form.select_auto_tag];
                  copy[idx] = { ...copy[idx], value: e.target.value };
                  update('select_auto_tag', '', copy);
                }}
              />
              {!disabled && (
                <RemoveBtn
                  type="button"
                  onClick={() => {
                    const copy = form.select_auto_tag.filter((_, i) => i !== idx);
                    update('select_auto_tag', '', copy);
                  }}
                >
                  &times;
                </RemoveBtn>
              )}
            </AutoTagRow>
          ))}
          {!disabled && (
            <AddBtn
              type="button"
              onClick={() => {
                const copy = [...form.select_auto_tag, { key: '', value: '', logic: 'Exists' }];
                update('select_auto_tag', '', copy);
              }}
            >
              + Add Auto Tag
            </AddBtn>
          )}
        </div>
      </OverlayTipRight>
      {errors.auto_tag && <AlertBanner>{errors.auto_tag}</AlertBanner>}
      <SubtitleRow>
        <Label>Group Permissions</Label>
      </SubtitleRow>
      <OverlayTipRight tip={TOOLTIPS.groups}>
        <SelectGroups
          groups={form.select_groups}
          disabled={disabled}
          setGroups={(groups: Record<string, boolean>) => update('select_groups', '', groups)}
        />
      </OverlayTipRight>
    </ImageFieldsWrapper>
  );
};

const OutputCollection: React.FC<OutputCollectionProps> = ({ value, onChange, onValidate, groups, mode, disabled = false, resetKey }) => {
  const [form, setFormState] = useState<FormOutputCollection>(() => apiToForm(value ?? {}, groups));
  // Re-derive the internal form from value when the parent signals a fresh dataset
  // (e.g. after a save refetch), without clobbering in-progress edits.
  const [prevResetKey, setPrevResetKey] = useState(resetKey);
  if (resetKey !== prevResetKey) {
    setPrevResetKey(resetKey);
    setFormState(apiToForm(value ?? {}, groups));
  }
  const errors = useMemo(() => validateOC(form), [form]);

  useEffect(() => {
    onValidate?.(Object.keys(errors).length > 0);
  }, [errors]);

  const setForm = (newForm: FormOutputCollection) => {
    setFormState(newForm);
    onChange(formToApi(newForm));
  };

  if (mode === ImageFormMode.View) {
    return (
      <>
        <SectionRow>
          <div>
            <OverlayTipRight tip={TOOLTIPS.self}>
              <b>Output Collection</b>
            </OverlayTipRight>
          </div>
        </SectionRow>
        <DisplayOutputCollection value={value} />
      </>
    );
  }

  const isEdit = mode === ImageFormMode.Edit;

  if (isEdit) {
    return (
      <SectionRow>
        <EditSpacer />
        <EditMiddle>
          <OverlayTipRight tip={TOOLTIPS.self}>
            <b>Output Collection</b>
          </OverlayTipRight>
        </EditMiddle>
        <EditFieldCol>
          <OutputCollectionInputs form={form} setForm={setForm} errors={errors} disabled={disabled} />
        </EditFieldCol>
      </SectionRow>
    );
  }

  return (
    <SectionRow>
      <TitleCol>
        <h5>Output Collection</h5>
      </TitleCol>
      <FieldCol>
        <OutputCollectionInputs form={form} setForm={setForm} errors={errors} disabled={disabled} />
      </FieldCol>
    </SectionRow>
  );
};

export default OutputCollection;
