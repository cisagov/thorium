import React, { useState, useMemo, useEffect } from 'react';
import styled from 'styled-components';

// project imports
import {
  SectionRow,
  ValCol,
  EditSpacer,
  EditMiddle,
  EditFieldCol,
  FieldGroup,
  Label,
  SwitchRow,
  ImageFieldsWrapper,
} from './shared.styled';
import { ImageFormMode } from './types';
import Markdown from '@components/shared/syntax/Markdown';
import FieldBadge from '@components/shared/badges/FieldBadge';
import { FieldError, errorOutline } from '@components/shared/inputs/FieldError';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import type { ImageVersion, ImageLifetime, SpawnLimitsValue } from '@models/images';
import { ImageScaler } from '@models/images';
import { OutputDisplayType } from '@models/results';
import type { SemVer } from '@models/semver';

const TOOLTIPS = {
  name: `Image name that contains only alpha-numeric characters and dashes.`,
  creator: `The user that created this image.`,
  group: `The Thorium group that can use this image.`,
  version: `The version of this image. Can be a semver string (e.g., 1.0.0) or a custom version string.`,
  description: `A description of this image's purpose and functionality.`,
  scaler: `The scaler type that executes this image. The scaler determines where Thorium will execute your tool. You must have the developer role permission for this scaler.`,
  image: `The container registry path:tag for this K8s scaled image.`,
  timeout: `The max time in seconds that an image will be allowed to run before it is terminated.`,
  lifetime: `How long k8s scaled pods of this image will run before being scaled down. Units are time in seconds or number of jobs.`,
  runtime: `Average execution time for the previous 100k runs of this image. (600 default).`,
  display_type: `The format used to render image results (JSON, String, etc). Results files are not rendered, download links for those will be shown.`,
  spawn_limit: `The max number of tool instances that can be run simultaneously. This is useful when tools interact with a performance limited API or database to prevent overloading that resource.`,
  collect_logs: `Whether the Thorium agent collects stdout/err as logs when this image runs.`,
  generator: `Whether this image is a Thorium generator that will be respawned until it completes creating jobs.`,
  used_by: `The pipelines that use this image. You cannot delete an image that is used by a pipeline.`,
};

const DISPLAY_TYPES: OutputDisplayType[] = [
  OutputDisplayType.Json,
  OutputDisplayType.String,
  OutputDisplayType.Table,
  OutputDisplayType.Markdown,
  OutputDisplayType.Xml,
  OutputDisplayType.Html,
  OutputDisplayType.Image,
  OutputDisplayType.Disassembly,
  OutputDisplayType.Hidden,
  OutputDisplayType.Custom,
];

const SCALER_TYPES: ImageScaler[] = [ImageScaler.K8s, ImageScaler.BareMetal, ImageScaler.External, ImageScaler.Windows, ImageScaler.Kvm];

const KeyCol = styled.div`
  flex: 0 0 auto;
  min-width: 140px;
`;

const Input = styled.input<{ $error?: boolean }>`
  width: 100%;
  padding: 6px 12px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  font-size: 14px;
  ${({ $error }) => $error && errorOutline}

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const TextArea = styled.textarea`
  width: 100%;
  padding: 6px 12px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  font-size: 14px;
  resize: vertical;
`;

const Select = styled.select<{ $error?: boolean }>`
  width: 100%;
  padding: 6px 12px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  font-size: 14px;
  ${({ $error }) => $error && errorOutline}
`;

const LifetimeRow = styled.div`
  display: flex;
  gap: 8px;
`;

const LifetimeAmountCol = styled.div`
  flex: 2;
`;

const LifetimeUnitCol = styled.div`
  flex: 1;
`;

const SwitchLabel = styled.span`
  font-size: 13px;
  font-weight: 600;
  color: var(--thorium-secondary-text);
`;

const ToggleTrack = styled.label<{ $checked: boolean; $disabled: boolean }>`
  position: relative;
  display: inline-block;
  width: 36px;
  height: 20px;
  border-radius: 10px;
  background: ${({ $checked }) => ($checked ? 'var(--thorium-ok-bg, #198754)' : 'var(--thorium-panel-border, #6c757d)')};
  cursor: ${({ $disabled }) => ($disabled ? 'not-allowed' : 'pointer')};
  opacity: ${({ $disabled }) => ($disabled ? 0.5 : 1)};
  transition: background 0.2s;

  input {
    opacity: 0;
    width: 0;
    height: 0;
    position: absolute;
  }
`;

const ToggleThumb = styled.span<{ $checked: boolean }>`
  position: absolute;
  top: 2px;
  left: ${({ $checked }) => ($checked ? '18px' : '2px')};
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #fff;
  transition: left 0.2s;
`;

export interface FieldsValue {
  name: string;
  group: string;
  version?: ImageVersion;
  description?: string;
  scaler: ImageScaler;
  image?: string;
  timeout?: number | string;
  lifetime?: ImageLifetime;
  display_type?: OutputDisplayType | string;
  spawn_limit: SpawnLimitsValue;
  collect_logs: boolean;
  generator: boolean;
}

interface FormFields {
  name: string;
  group: string;
  version_text: string;
  description: string;
  scaler: ImageScaler;
  image: string;
  timeout: string;
  lifetime_amount: string;
  lifetime_counter: string;
  display_type: string;
  spawn_limit_text: string;
  collect_logs: boolean;
  generator: boolean;
}

interface FieldsProps {
  value: FieldsValue;
  onChange: (value: FieldsValue) => void;
  onValidate?: (hasErrors: boolean) => void;
  groups: string[];
  mode: ImageFormMode;
  showErrors?: boolean;
  creator?: string;
  runtime?: number;
  usedBy?: string[];
  resetKey?: number;
}

function getSemVerString(version: SemVer): string {
  let s = `${version.major}.${version.minor}.${version.patch}`;
  if (version.pre) s += `-${version.pre}`;
  if (version.build) s += `+${version.build}`;
  return s;
}

function formatLifetimeDisplay(lifetime?: ImageLifetime): string | undefined {
  if (!lifetime) return undefined;
  if (lifetime.counter === 'time') return `${lifetime.amount} Seconds`;
  if (lifetime.counter === 'jobs') return `${lifetime.amount} Jobs`;
  return undefined;
}

function apiToForm(value: FieldsValue): FormFields {
  let versionText = '';
  if (value.version) {
    if (value.version.Custom) {
      versionText = value.version.Custom;
    } else if (value.version.SemVer) {
      versionText = getSemVerString(value.version.SemVer);
    }
  }

  let spawnLimitText = '';
  if (value.spawn_limit !== 'Unlimited' && typeof value.spawn_limit === 'object' && 'Basic' in value.spawn_limit) {
    spawnLimitText = String(value.spawn_limit.Basic);
  }

  return {
    name: value.name ?? '',
    group: value.group ?? '',
    version_text: versionText,
    description: value.description && value.description !== 'null' ? value.description : '',
    scaler: value.scaler ?? ImageScaler.K8s,
    image: value.image ?? '',
    timeout: value.timeout != null ? String(value.timeout) : '',
    lifetime_amount: value.lifetime?.amount != null ? String(value.lifetime.amount) : '',
    lifetime_counter: value.lifetime?.counter ?? 'jobs',
    display_type: typeof value.display_type === 'string' ? value.display_type : '',
    spawn_limit_text: spawnLimitText,
    collect_logs: value.collect_logs ?? true,
    generator: value.generator ?? false,
  };
}

function formToApi(form: FormFields): FieldsValue {
  const result: FieldsValue = {
    name: form.name,
    group: form.group,
    scaler: form.scaler,
    collect_logs: form.collect_logs,
    generator: form.generator,
    spawn_limit: 'Unlimited',
  };

  if (form.version_text.trim()) {
    result.version = { Custom: form.version_text.trim() };
  }

  const desc = form.description.trim();
  if (desc) result.description = desc;

  if (form.scaler === ImageScaler.K8s) {
    result.image = form.image;
  }

  if (form.timeout && !isNaN(Number(form.timeout))) {
    result.timeout = Number(form.timeout);
  }

  if (form.scaler === ImageScaler.K8s && form.lifetime_amount) {
    result.lifetime = {
      counter: form.lifetime_counter,
      amount: Number(form.lifetime_amount),
    };
  }

  if (form.display_type) {
    result.display_type = form.display_type;
  }

  if (form.spawn_limit_text) {
    result.spawn_limit = { Basic: Number(form.spawn_limit_text) };
  }

  return result;
}

function validateFields(form: FormFields): Record<string, string> {
  const errors: Record<string, string> = {};

  if (!form.name) errors.name = 'Required';
  if (!form.group) errors.group = 'Required';
  if (form.scaler === ImageScaler.K8s && !form.image) errors.image = 'Required';
  if (!form.display_type) errors.display_type = 'Required';

  if (form.scaler !== ImageScaler.External) {
    if (!form.timeout) {
      errors.timeout = 'Required';
    } else if (isNaN(Number(form.timeout))) {
      errors.timeout = 'Timeout must be an integer number (of seconds)';
    }
  }

  return errors;
}

// View mode: read-only display of all fields
const DisplayFields: React.FC<{
  value: FieldsValue;
  creator?: string;
  runtime?: number;
  usedBy?: string[];
}> = ({ value, creator, runtime, usedBy }) => {
  const spawnLimitDisplay =
    value.spawn_limit === 'Unlimited'
      ? 'Unlimited'
      : typeof value.spawn_limit === 'object' && 'Basic' in value.spawn_limit
        ? value.spawn_limit.Basic
        : value.spawn_limit;

  return (
    <>
      <SectionRow>
        <KeyCol>
          <OverlayTipRight tip={TOOLTIPS.creator}>
            <b>Creator</b>
          </OverlayTipRight>
        </KeyCol>
        <ValCol>
          <FieldBadge field={creator} color="#305ef2" />
        </ValCol>
      </SectionRow>
      <SectionRow>
        <KeyCol>
          <OverlayTipRight tip={TOOLTIPS.group}>
            <b>Group</b>
          </OverlayTipRight>
        </KeyCol>
        <ValCol>
          <FieldBadge field={value.group} color="#6a00db" />
        </ValCol>
      </SectionRow>
      {value.version && (
        <SectionRow>
          <KeyCol>
            <OverlayTipRight tip={TOOLTIPS.version}>
              <b>Version</b>
            </OverlayTipRight>
          </KeyCol>
          <ValCol>
            {value.version.Custom && <FieldBadge field={value.version.Custom} color="#7e7c7c" />}
            {value.version.SemVer && <FieldBadge field={getSemVerString(value.version.SemVer)} color="#7e7c7c" />}
          </ValCol>
        </SectionRow>
      )}
      <SectionRow>
        <KeyCol>
          <OverlayTipRight tip={TOOLTIPS.description}>
            <b>Description</b>
          </OverlayTipRight>
        </KeyCol>
        <ValCol>
          <Markdown>{value.description && value.description !== 'null' ? value.description : ''}</Markdown>
        </ValCol>
      </SectionRow>
      <SectionRow>
        <KeyCol>
          <OverlayTipRight tip={TOOLTIPS.scaler}>
            <b>Scaler</b>
          </OverlayTipRight>
        </KeyCol>
        <ValCol>
          <FieldBadge field={value.scaler} color="#7e7c7c" />
        </ValCol>
      </SectionRow>
      {value.scaler === ImageScaler.K8s && (
        <SectionRow>
          <KeyCol>
            <OverlayTipRight tip={TOOLTIPS.image}>
              <b>Image</b>
            </OverlayTipRight>
          </KeyCol>
          <ValCol>
            <FieldBadge field={value.image} color="#7e7c7c" />
          </ValCol>
        </SectionRow>
      )}
      <SectionRow>
        <KeyCol>
          <OverlayTipRight tip={TOOLTIPS.timeout}>
            <b>Timeout</b>
          </OverlayTipRight>
        </KeyCol>
        <ValCol>
          <FieldBadge field={value.timeout} color="#7e7c7c" />
        </ValCol>
      </SectionRow>
      {value.scaler === ImageScaler.K8s && (
        <SectionRow>
          <KeyCol>
            <OverlayTipRight tip={TOOLTIPS.lifetime}>
              <b>Lifetime</b>
            </OverlayTipRight>
          </KeyCol>
          <ValCol>
            <FieldBadge field={formatLifetimeDisplay(value.lifetime)} color="#7e7c7c" />
          </ValCol>
        </SectionRow>
      )}
      <SectionRow>
        <KeyCol>
          <OverlayTipRight tip={TOOLTIPS.runtime}>
            <b>Runtime</b>
          </OverlayTipRight>
        </KeyCol>
        <ValCol>
          <FieldBadge field={runtime} color="#7e7c7c" />
        </ValCol>
      </SectionRow>
      <SectionRow>
        <KeyCol>
          <OverlayTipRight tip={TOOLTIPS.display_type}>
            <b>Display Type</b>
          </OverlayTipRight>
        </KeyCol>
        <ValCol>
          <FieldBadge field={value.display_type} color="#7e7c7c" />
        </ValCol>
      </SectionRow>
      <SectionRow>
        <KeyCol>
          <OverlayTipRight tip={TOOLTIPS.spawn_limit}>
            <b>Spawn Limit</b>
          </OverlayTipRight>
        </KeyCol>
        <ValCol>
          <FieldBadge field={spawnLimitDisplay} color="#7e7c7c" />
        </ValCol>
      </SectionRow>
      <SectionRow>
        <KeyCol>
          <OverlayTipRight tip={TOOLTIPS.collect_logs}>
            <b>Logging Enabled</b>
          </OverlayTipRight>
        </KeyCol>
        <ValCol>
          <FieldBadge field={value.collect_logs} color="#7e7c7c" />
        </ValCol>
      </SectionRow>
      <SectionRow>
        <KeyCol>
          <OverlayTipRight tip={TOOLTIPS.generator}>
            <b>Generator</b>
          </OverlayTipRight>
        </KeyCol>
        <ValCol>
          <FieldBadge field={value.generator} color="#7e7c7c" />
        </ValCol>
      </SectionRow>
      <SectionRow>
        <KeyCol>
          <OverlayTipRight tip={TOOLTIPS.used_by}>
            <b>Used By</b>
          </OverlayTipRight>
        </KeyCol>
        <ValCol>
          <FieldBadge field={usedBy} color="#7e7c7c" />
        </ValCol>
      </SectionRow>
    </>
  );
};

// Editable fields form shared between Create/Copy/Edit modes
const FieldInputs: React.FC<{
  form: FormFields;
  setForm: (f: FormFields) => void;
  errors: Record<string, string>;
  groups: string[];
  mode: ImageFormMode;
  showErrors: boolean;
  creator?: string;
  runtime?: number;
  usedBy?: string[];
}> = ({ form, setForm, errors, groups, mode, showErrors, creator, runtime, usedBy }) => {
  const update = (key: keyof FormFields, val: FormFields[keyof FormFields]) => {
    setForm({ ...form, [key]: val });
  };

  const isEdit = mode === ImageFormMode.Edit;
  const descriptionHeight = form.description ? Math.max(form.description.split(/\r\n|\r|\n/).length * 25, 200) : 200;

  return (
    <ImageFieldsWrapper>
      {/* Creator - read-only in Edit, not shown in Create/Copy */}
      {isEdit && (
        <FieldGroup>
          <Label>Creator</Label>
          <FieldBadge field={creator} color="#305ef2" />
        </FieldGroup>
      )}

      {/* Name - editable in Create/Copy, not shown in Edit */}
      {!isEdit && (
        <FieldGroup>
          <Label>Name</Label>
          <OverlayTipRight tip={TOOLTIPS.name}>
            <Input
              type="text"
              value={form.name}
              placeholder="name"
              $error={showErrors && !!errors.name}
              onChange={(e) => update('name', e.target.value)}
            />
          </OverlayTipRight>
          {errors.name && showErrors && <FieldError>{errors.name}</FieldError>}
        </FieldGroup>
      )}

      {/* Group - editable in Create/Copy, read-only in Edit */}
      {isEdit ? (
        <FieldGroup>
          <Label>Group</Label>
          <FieldBadge field={form.group} color="#6a00db" />
        </FieldGroup>
      ) : (
        <FieldGroup>
          <Label>Group</Label>
          <OverlayTipRight tip={TOOLTIPS.group}>
            <Select value={form.group} $error={showErrors && !!errors.group} onChange={(e) => update('group', e.target.value)}>
              <option value="">Select a group</option>
              {groups.sort().map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </Select>
          </OverlayTipRight>
          {errors.group && showErrors && <FieldError>{errors.group}</FieldError>}
        </FieldGroup>
      )}

      {/* Description */}
      <FieldGroup>
        <Label>Description</Label>
        <OverlayTipRight tip={TOOLTIPS.description}>
          <TextArea
            style={{ minHeight: `${descriptionHeight}px` }}
            value={form.description}
            placeholder="describe this image"
            onChange={(e) => update('description', e.target.value)}
          />
        </OverlayTipRight>
      </FieldGroup>

      {/* Version */}
      <FieldGroup>
        <Label>Version</Label>
        <OverlayTipRight tip={TOOLTIPS.version}>
          <Input
            type="text"
            value={form.version_text}
            placeholder="1.0.0"
            onChange={(e) => update('version_text', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>

      {/* Scaler */}
      <FieldGroup>
        <Label>Scaler</Label>
        <OverlayTipRight tip={TOOLTIPS.scaler}>
          <Select value={form.scaler} onChange={(e) => update('scaler', e.target.value)}>
            {SCALER_TYPES.sort().map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
        </OverlayTipRight>
      </FieldGroup>

      {/* Image/Tag - only for K8s */}
      <FieldGroup>
        <Label>Image/Tag</Label>
        <OverlayTipRight tip={TOOLTIPS.image}>
          <Input
            type="text"
            value={form.image}
            disabled={form.scaler !== ImageScaler.K8s}
            placeholder="docker:latest"
            $error={showErrors && !!errors.image}
            onChange={(e) => update('image', e.target.value.trim())}
          />
        </OverlayTipRight>
        {errors.image && showErrors && <FieldError>{errors.image}</FieldError>}
      </FieldGroup>

      {/* Timeout */}
      <FieldGroup>
        <Label>Timeout (seconds)</Label>
        <OverlayTipRight tip={TOOLTIPS.timeout}>
          <Input
            type="text"
            value={form.timeout}
            disabled={form.scaler === ImageScaler.External}
            placeholder="seconds"
            $error={showErrors && !!errors.timeout}
            onChange={(e) => {
              const val = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
              update('timeout', val);
            }}
          />
        </OverlayTipRight>
        {errors.timeout && showErrors && <FieldError>{errors.timeout}</FieldError>}
      </FieldGroup>

      {/* Lifetime - only for K8s */}
      {form.scaler === ImageScaler.K8s && (
        <FieldGroup>
          <Label>Lifetime</Label>
          <OverlayTipRight tip={TOOLTIPS.lifetime}>
            <LifetimeRow>
              <LifetimeAmountCol>
                <Input
                  type="text"
                  value={form.lifetime_amount}
                  placeholder="0"
                  onChange={(e) => {
                    const val = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
                    update('lifetime_amount', val);
                  }}
                />
              </LifetimeAmountCol>
              <LifetimeUnitCol>
                <Select value={form.lifetime_counter} onChange={(e) => update('lifetime_counter', e.target.value)}>
                  <option value="jobs">Jobs</option>
                  <option value="time">Seconds</option>
                </Select>
              </LifetimeUnitCol>
            </LifetimeRow>
          </OverlayTipRight>
        </FieldGroup>
      )}

      {/* Runtime - read-only in Edit */}
      {isEdit && (
        <FieldGroup>
          <Label>Runtime</Label>
          <FieldBadge field={runtime} color="#7e7c7c" />
        </FieldGroup>
      )}

      {/* Display Type */}
      <FieldGroup>
        <Label>Display Type</Label>
        <OverlayTipRight tip={TOOLTIPS.display_type}>
          <Select
            value={form.display_type}
            $error={showErrors && !!errors.display_type}
            onChange={(e) => update('display_type', e.target.value)}
          >
            <option value="">Select a display type</option>
            {DISPLAY_TYPES.map((dt) => (
              <option key={dt} value={dt}>
                {dt}
              </option>
            ))}
          </Select>
        </OverlayTipRight>
        {errors.display_type && showErrors && <FieldError>{errors.display_type}</FieldError>}
      </FieldGroup>

      {/* Spawn Limit */}
      <FieldGroup>
        <Label>Spawn Limit</Label>
        <OverlayTipRight tip={TOOLTIPS.spawn_limit}>
          <Input
            type="text"
            value={form.spawn_limit_text}
            placeholder="Unlimited"
            onChange={(e) => {
              const val = e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '';
              update('spawn_limit_text', val);
            }}
          />
        </OverlayTipRight>
      </FieldGroup>

      {/* Collect Logs */}
      <SwitchRow>
        <SwitchLabel>Collect Logs</SwitchLabel>
        <OverlayTipRight tip={TOOLTIPS.collect_logs}>
          <ToggleTrack $checked={form.collect_logs} $disabled={form.scaler === ImageScaler.External}>
            <input
              type="checkbox"
              checked={form.collect_logs}
              disabled={form.scaler === ImageScaler.External}
              onChange={() => update('collect_logs', !form.collect_logs)}
            />
            <ToggleThumb $checked={form.collect_logs} />
          </ToggleTrack>
        </OverlayTipRight>
      </SwitchRow>

      {/* Generator */}
      <SwitchRow>
        <SwitchLabel>Generator</SwitchLabel>
        <OverlayTipRight tip={TOOLTIPS.generator}>
          <ToggleTrack $checked={form.generator} $disabled={form.scaler === ImageScaler.External}>
            <input
              type="checkbox"
              checked={form.generator}
              disabled={form.scaler === ImageScaler.External}
              onChange={() => update('generator', !form.generator)}
            />
            <ToggleThumb $checked={form.generator} />
          </ToggleTrack>
        </OverlayTipRight>
      </SwitchRow>

      {/* Used By - read-only in Edit */}
      {isEdit && (
        <FieldGroup>
          <Label>Used By</Label>
          <FieldBadge field={usedBy} color="#7e7c7c" />
        </FieldGroup>
      )}
    </ImageFieldsWrapper>
  );
};

const Fields: React.FC<FieldsProps> = ({
  value,
  onChange,
  onValidate,
  groups,
  mode,
  showErrors = false,
  creator,
  runtime,
  usedBy,
  resetKey,
}) => {
  const [form, setFormState] = useState<FormFields>(() => apiToForm(value));
  // Re-derive the internal form from value when the parent signals a fresh dataset
  // (e.g. after a save refetch), without clobbering in-progress edits.
  const [prevResetKey, setPrevResetKey] = useState(resetKey);
  if (resetKey !== prevResetKey) {
    setPrevResetKey(resetKey);
    setFormState(apiToForm(value));
  }
  const errors = useMemo(() => validateFields(form), [form]);

  useEffect(() => {
    onValidate?.(Object.keys(errors).length > 0);
  }, [errors]);

  const setForm = (newForm: FormFields) => {
    setFormState(newForm);
    onChange(formToApi(newForm));
  };

  if (mode === ImageFormMode.View) {
    return <DisplayFields value={value} creator={creator} runtime={runtime} usedBy={usedBy} />;
  }

  const isEdit = mode === ImageFormMode.Edit;

  if (isEdit) {
    return (
      <SectionRow>
        <EditSpacer />
        <EditMiddle>
          <OverlayTipRight tip="Core image fields.">
            <b>Image Fields</b>
          </OverlayTipRight>
        </EditMiddle>
        <EditFieldCol>
          <FieldInputs
            form={form}
            setForm={setForm}
            errors={errors}
            groups={groups}
            mode={mode}
            showErrors={showErrors}
            creator={creator}
            runtime={runtime}
            usedBy={usedBy}
          />
        </EditFieldCol>
      </SectionRow>
    );
  }

  return <FieldInputs form={form} setForm={setForm} errors={errors} groups={groups} mode={mode} showErrors={showErrors} />;
};

export default Fields;
