import React, { useEffect, useMemo, useState } from 'react';

// project imports
import {
  EditFieldCol,
  EditMiddle,
  EditSpacer,
  FieldGroup,
  Input,
  Label,
  PipelineFieldsWrapper,
  SectionRow,
  Select,
  TextArea,
} from './shared.styled';
import { PipelineFormMode } from './types';
import FieldBadge from '@components/shared/badges/FieldBadge';
import { FieldError } from '@components/shared/inputs/FieldError';
import Markdown from '@components/shared/syntax/Markdown';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import { DEFAULT_SLA } from '@utilities/transforms/pipeline';

const TOOLTIPS = {
  name: `Pipeline name that contains only alpha-numeric characters and dashes.`,
  group: `The Thorium group that owns this pipeline. Images run within a pipeline must be in the same group.`,
  description: `A description of this pipeline's purpose. Supports markdown.`,
  sla: `The number of seconds Thorium has to meet this pipeline's SLA.`,
};

/// The subset of pipeline fields edited by this form (name, group, description, SLA)
export interface PipelineFieldsValue {
  /// The pipeline name
  name: string;
  /// The group that owns the pipeline
  group: string;
  /// The optional markdown description
  description?: string;
  /// The SLA in seconds
  sla?: number;
}

/// The internal string-backed form state for the fields
interface FormFields {
  /// The pipeline name
  name: string;
  /// The group that owns the pipeline
  group: string;
  /// The description text
  description: string;
  /// The SLA as raw input text (digits only)
  sla: string;
}

interface FieldsProps {
  /// The current pipeline field values
  value: PipelineFieldsValue;
  /// Called with the updated field values whenever the user edits a field
  onChange: (value: PipelineFieldsValue) => void;
  /// Called with whether the fields currently have validation errors
  onValidate?: (hasErrors: boolean) => void;
  /// The groups the user can create pipelines in (used for the group select)
  groups: string[];
  /// The mode this form is rendered in
  mode: PipelineFormMode;
  /// Whether to surface validation errors to the user
  showErrors?: boolean;
  /// Increment to force the internal form to re-derive from `value` (e.g. after a refetch)
  resetKey?: number;
}

/// Convert the API-shaped field values into the string-backed form state
function apiToForm(value: PipelineFieldsValue): FormFields {
  return {
    name: value.name ?? '',
    group: value.group ?? '',
    description: value.description && value.description !== 'null' ? value.description : '',
    sla: value.sla != null ? String(value.sla) : '',
  };
}

/// Convert the string-backed form state back into API-shaped field values
function formToApi(form: FormFields): PipelineFieldsValue {
  const result: PipelineFieldsValue = {
    name: form.name,
    group: form.group,
  };
  const desc = form.description.trim();
  if (desc) result.description = desc;
  if (form.sla && !isNaN(Number(form.sla))) result.sla = Number(form.sla);
  return result;
}

/// Compute validation errors for the required fields
function validateFields(form: FormFields, mode: PipelineFormMode): Record<string, string> {
  const errors: Record<string, string> = {};
  if (!form.name) errors.name = 'Required';
  if (!form.group) errors.group = 'Required';
  // On create/copy the SLA may be left blank — it defaults to DEFAULT_SLA on submit (shown as
  // placeholder). In edit it must stay set. A provided value must always be a valid integer.
  if (mode === PipelineFormMode.Edit && !form.sla) {
    errors.sla = 'Required';
  } else if (form.sla && isNaN(Number(form.sla))) {
    errors.sla = 'SLA must be an integer number (of seconds)';
  }
  return errors;
}

/// Read-only display of the pipeline fields used in View mode
const DisplayFields: React.FC<{ value: PipelineFieldsValue }> = ({ value }) => (
  <>
    <SectionRow>
      <FieldGroup>
        <Label>Group</Label>
        <FieldBadge field={value.group} color="#6a00db" />
      </FieldGroup>
    </SectionRow>
    <SectionRow>
      <FieldGroup>
        <Label>Description</Label>
        <Markdown>{value.description && value.description !== 'null' ? value.description : ''}</Markdown>
      </FieldGroup>
    </SectionRow>
    <SectionRow>
      <FieldGroup>
        <Label>SLA (seconds)</Label>
        <FieldBadge field={value.sla} color="#7e7c7c" />
      </FieldGroup>
    </SectionRow>
  </>
);

/// The editable field inputs shared between Create/Copy/Edit modes
const FieldInputs: React.FC<{
  form: FormFields;
  setForm: (f: FormFields) => void;
  errors: Record<string, string>;
  groups: string[];
  mode: PipelineFormMode;
  showErrors: boolean;
}> = ({ form, setForm, errors, groups, mode, showErrors }) => {
  // update a single field and propagate the change upward
  const update = (key: keyof FormFields, val: string) => setForm({ ...form, [key]: val });
  // name and group are immutable once a pipeline exists
  const isEdit = mode === PipelineFormMode.Edit;
  const descriptionHeight = form.description ? Math.max(form.description.split(/\r\n|\r|\n/).length * 25, 150) : 150;
  return (
    <PipelineFieldsWrapper>
      {/* Name - editable in Create/Copy, locked in Edit */}
      <FieldGroup>
        <Label>Name</Label>
        {isEdit ? (
          <FieldBadge field={form.name} color="#7e7c7c" />
        ) : (
          <>
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
          </>
        )}
      </FieldGroup>

      {/* Group - editable in Create/Copy, locked in Edit */}
      <FieldGroup>
        <Label>Group</Label>
        {isEdit ? (
          <FieldBadge field={form.group} color="#6a00db" />
        ) : (
          <>
            <OverlayTipRight tip={TOOLTIPS.group}>
              <Select value={form.group} $error={showErrors && !!errors.group} onChange={(e) => update('group', e.target.value)}>
                <option value="">Select a group</option>
                {[...groups].sort().map((g) => (
                  <option key={g} value={g}>
                    {g}
                  </option>
                ))}
              </Select>
            </OverlayTipRight>
            {errors.group && showErrors && <FieldError>{errors.group}</FieldError>}
          </>
        )}
      </FieldGroup>

      {/* Description */}
      <FieldGroup>
        <Label>Description</Label>
        <OverlayTipRight tip={TOOLTIPS.description} block>
          <TextArea
            style={{ minHeight: `${descriptionHeight}px` }}
            value={form.description}
            placeholder="describe this pipeline"
            onChange={(e) => update('description', e.target.value)}
          />
        </OverlayTipRight>
      </FieldGroup>

      {/* SLA */}
      <FieldGroup>
        <Label>SLA (seconds)</Label>
        <OverlayTipRight tip={TOOLTIPS.sla}>
          <Input
            type="text"
            value={form.sla}
            placeholder={String(DEFAULT_SLA)}
            $error={showErrors && !!errors.sla}
            onChange={(e) => update('sla', e.target.value ? e.target.value.replace(/[^0-9]+/gi, '') : '')}
          />
        </OverlayTipRight>
        {errors.sla && showErrors && <FieldError>{errors.sla}</FieldError>}
      </FieldGroup>
    </PipelineFieldsWrapper>
  );
};

/// Pipeline core fields form (name, group, description, SLA) supporting create, copy, edit, and view
const Fields: React.FC<FieldsProps> = ({ value, onChange, onValidate, groups, mode, showErrors = false, resetKey }) => {
  const [form, setFormState] = useState<FormFields>(() => apiToForm(value));
  // Re-derive the internal form from value when the parent signals a fresh dataset
  // (e.g. after a save refetch), without clobbering in-progress edits.
  const [prevResetKey, setPrevResetKey] = useState(resetKey);
  if (resetKey !== prevResetKey) {
    setPrevResetKey(resetKey);
    setFormState(apiToForm(value));
  }
  const errors = useMemo(() => validateFields(form, mode), [form, mode]);
  useEffect(() => {
    onValidate?.(Object.keys(errors).length > 0);
  }, [errors]);
  // update internal state and emit the API-shaped value upward
  const setForm = (newForm: FormFields) => {
    setFormState(newForm);
    onChange(formToApi(newForm));
  };
  if (mode === PipelineFormMode.View) {
    return <DisplayFields value={value} />;
  }
  if (mode === PipelineFormMode.Edit) {
    return (
      <SectionRow>
        <EditSpacer />
        <EditMiddle>
          <OverlayTipRight tip="Core pipeline fields.">
            <b>Pipeline</b>
          </OverlayTipRight>
        </EditMiddle>
        <EditFieldCol>
          <FieldInputs form={form} setForm={setForm} errors={errors} groups={groups} mode={mode} showErrors={showErrors} />
        </EditFieldCol>
      </SectionRow>
    );
  }
  return <FieldInputs form={form} setForm={setForm} errors={errors} groups={groups} mode={mode} showErrors={showErrors} />;
};

export default Fields;
