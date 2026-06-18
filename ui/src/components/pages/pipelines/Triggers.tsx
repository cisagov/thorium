import React, { useEffect, useState } from 'react';
import { FaTrash } from 'react-icons/fa';

// project imports
import { Input, Label, Select } from './shared.styled';
import TriggerDisplay from './TriggerDisplay';
import {
  AddButton,
  ButtonSpacer,
  CardHeader,
  DeleteCol,
  EmptyNote,
  FilterRow,
  FilterSection,
  HeaderField,
  IconButton,
  Panel,
  TriggerCard,
} from './Triggers.styled';
import { PipelineFormMode } from './types';
import { FieldError } from '@components/shared/inputs/FieldError';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import type { EventTrigger, TagTrigger } from '@models/pipelines';
import { TagTypes } from '@models/tags';

/// Section help for the Triggers content. Co-located with the shared Triggers editor (used by both
/// the create and edit views) so the text is single-source and can't drift between them.
const TRIGGERS_TIP = 'Automatic triggers that cause this pipeline to run on new samples or matching tags.';

/// The kind of event trigger the user is configuring
export enum TriggerKind {
  /// Trigger when a new sample is uploaded
  NewSample = 'NewSample',
  /// Trigger when tags matching a filter are created
  Tag = 'Tag',
}

/// A single key/value row used to build a tag filter map
export interface KeyValueEntry {
  /// The tag key
  key: string;
  /// One value for the tag key
  value: string;
}

/// The editable form state for a single named trigger
export interface FormTrigger {
  /// The unique name of this trigger
  name: string;
  /// The kind of trigger (NewSample or Tag)
  kind: TriggerKind;
  /// The tag types this trigger watches (only used for Tag triggers)
  tagTypes: TagTypes[];
  /// The required tag key/value rows (only used for Tag triggers)
  required: KeyValueEntry[];
  /// The excluded ("not") tag key/value rows (only used for Tag triggers)
  not: KeyValueEntry[];
}

const ALL_TAG_TYPES: string[] = Object.values(TagTypes);

/// Convert a tag filter map into a flat list of key/value rows for editing
function mapToEntries(map: Record<string, string[]>): KeyValueEntry[] {
  const entries: KeyValueEntry[] = [];
  for (const key of Object.keys(map)) {
    for (const value of map[key]) {
      entries.push({ key, value });
    }
  }
  return entries;
}

/// Aggregate flat key/value rows back into a tag filter map, dropping empty rows
function entriesToMap(entries: KeyValueEntry[]): Record<string, string[]> {
  const map: Record<string, string[]> = {};
  for (const { key, value } of entries) {
    const k = key.trim();
    const v = value.trim();
    if (!k || !v) continue;
    if (!map[k]) map[k] = [];
    if (!map[k].includes(v)) map[k].push(v);
  }
  return map;
}

/// Type guard for a Tag-kind event trigger
function isTagTrigger(trigger: EventTrigger): trigger is { Tag: TagTrigger } {
  return typeof trigger === 'object' && 'Tag' in trigger;
}

/// Convert the API trigger map into the editable form state list
export function triggersToForm(triggers: Record<string, EventTrigger>): FormTrigger[] {
  return Object.entries(triggers).map(([name, trigger]) => {
    if (isTagTrigger(trigger)) {
      return {
        name,
        kind: TriggerKind.Tag,
        tagTypes: trigger.Tag.tag_types ?? [],
        required: mapToEntries(trigger.Tag.required ?? {}),
        not: mapToEntries(trigger.Tag.not ?? {}),
      };
    }
    return { name, kind: TriggerKind.NewSample, tagTypes: [], required: [], not: [] };
  });
}

/// Convert the editable form state list back into the API trigger map
export function formToTriggers(forms: FormTrigger[]): Record<string, EventTrigger> {
  const triggers: Record<string, EventTrigger> = {};
  for (const form of forms) {
    const name = form.name.trim();
    if (!name) continue;
    if (form.kind === TriggerKind.Tag) {
      triggers[name] = {
        Tag: {
          tag_types: form.tagTypes,
          required: entriesToMap(form.required),
          not: entriesToMap(form.not),
        },
      };
    } else {
      triggers[name] = 'NewSample';
    }
  }
  return triggers;
}

/// Validate the trigger forms, requiring non-empty unique names and at least one tag type per Tag trigger
export function validateTriggers(forms: FormTrigger[]): boolean {
  const seen = new Set<string>();
  for (const form of forms) {
    const name = form.name.trim();
    if (!name) return true;
    if (seen.has(name)) return true;
    seen.add(name);
    if (form.kind === TriggerKind.Tag && form.tagTypes.length === 0) return true;
  }
  return false;
}

/// Compute a per-trigger name error message (or null), keyed by index, for displaying
/// the error directly under each name input.
export function triggerNameErrors(forms: FormTrigger[]): (string | null)[] {
  const counts = new Map<string, number>();
  for (const form of forms) {
    const name = form.name.trim();
    if (name) counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  return forms.map((form) => {
    const name = form.name.trim();
    if (!name) return 'Required';
    if ((counts.get(name) ?? 0) > 1) return 'Duplicate trigger name';
    return null;
  });
}

/// Create a blank Tag-kind trigger form entry
function blankTrigger(): FormTrigger {
  return { name: '', kind: TriggerKind.Tag, tagTypes: [], required: [], not: [] };
}

/// An editor for a single tag filter map (key/value rows with an auto-trailing-empty row).
/// The delete column is reserved on every row so the inputs keep a constant width.
const KeyValuesEditor: React.FC<{
  label: string;
  tip: string;
  entries: KeyValueEntry[];
  onChange: (entries: KeyValueEntry[]) => void;
}> = ({ label, tip, entries, onChange }) => {
  // always render a trailing empty row so a new pair can be added without a button
  const needsTrailing = entries.length === 0 || entries[entries.length - 1].key !== '' || entries[entries.length - 1].value !== '';
  const display = needsTrailing ? [...entries, { key: '', value: '' }] : entries;
  // update one row's key or value and prune fully-empty interior rows
  const updateRow = (idx: number, field: 'key' | 'value', val: string) => {
    const next = display.map((entry, i) => (i === idx ? { ...entry, [field]: val } : entry));
    onChange(next.filter((entry, i) => i === next.length - 1 || entry.key !== '' || entry.value !== ''));
  };
  const removeRow = (idx: number) => onChange(display.filter((_, i) => i !== idx && i !== display.length - 1));
  return (
    <FilterSection>
      <OverlayTipRight tip={tip}>
        <Label>{label}</Label>
      </OverlayTipRight>
      {display.map((entry, idx) => (
        <FilterRow key={idx}>
          <Input placeholder="tag key" value={entry.key} onChange={(e) => updateRow(idx, 'key', e.target.value)} />
          <Input placeholder="tag value" value={entry.value} onChange={(e) => updateRow(idx, 'value', e.target.value)} />
          {idx !== display.length - 1 ? (
            <IconButton type="button" onClick={() => removeRow(idx)} aria-label="Remove filter" title="Remove filter">
              <FaTrash />
            </IconButton>
          ) : (
            <ButtonSpacer aria-hidden />
          )}
        </FilterRow>
      ))}
    </FilterSection>
  );
};

/// A single editable trigger card (name, type, and — for Tag triggers — tag types + filters)
const TriggerCardItem: React.FC<{
  form: FormTrigger;
  nameError: string | null;
  showErrors: boolean;
  onChange: (patch: Partial<FormTrigger>) => void;
  onRemove: () => void;
}> = ({ form, nameError, showErrors, onChange, onRemove }) => {
  const tagTypesError = showErrors && form.kind === TriggerKind.Tag && form.tagTypes.length === 0;
  return (
    <TriggerCard>
      <CardHeader>
        <HeaderField>
          <Label>Trigger Name</Label>
          <Input placeholder="trigger name" value={form.name} $error={!!nameError} onChange={(e) => onChange({ name: e.target.value })} />
          {nameError && <FieldError>{nameError}</FieldError>}
        </HeaderField>
        <HeaderField>
          <Label>Type</Label>
          <Select value={form.kind} onChange={(e) => onChange({ kind: e.target.value as TriggerKind })}>
            <option value={TriggerKind.Tag}>Tag</option>
            <option value={TriggerKind.NewSample}>NewSample</option>
          </Select>
        </HeaderField>
        <DeleteCol>
          {/* hidden label keeps the button aligned with the inputs (which sit below their labels) */}
          <Label aria-hidden>&nbsp;</Label>
          <IconButton type="button" onClick={onRemove} aria-label="Remove trigger" title="Remove trigger">
            <FaTrash />
          </IconButton>
        </DeleteCol>
      </CardHeader>
      {form.kind === TriggerKind.Tag && (
        <>
          <FilterSection>
            <OverlayTipRight tip="The tag types this trigger watches (files, repos, or both).">
              <Label>Tag Types</Label>
            </OverlayTipRight>
            <SelectInputArray
              isCreatable={false}
              options={ALL_TAG_TYPES}
              values={form.tagTypes}
              onChange={(vals) => onChange({ tagTypes: vals as TagTypes[] })}
              defaultMessage="Select tag types..."
              error={tagTypesError}
            />
            {tagTypesError && <FieldError>Select at least one tag type</FieldError>}
          </FilterSection>
          <KeyValuesEditor
            label="Required"
            tip="Tags that must be set for this trigger to fire."
            entries={form.required}
            onChange={(entries) => onChange({ required: entries })}
          />
          <KeyValuesEditor
            label="Not"
            tip="Tags that, if set, prevent this trigger from firing."
            entries={form.not}
            onChange={(entries) => onChange({ not: entries })}
          />
        </>
      )}
    </TriggerCard>
  );
};

interface TriggersProps {
  /// The current trigger map keyed by trigger name
  value: Record<string, EventTrigger>;
  /// Called with the updated trigger map whenever the user edits a trigger
  onChange: (value: Record<string, EventTrigger>) => void;
  /// Called with whether the triggers currently have validation errors
  onValidate?: (hasErrors: boolean) => void;
  /// The mode this editor is rendered in
  mode: PipelineFormMode;
  /// Whether to surface per-field validation errors (set after a create/save attempt)
  showErrors?: boolean;
  /// Increment to force the internal form to re-derive from `value`
  resetKey?: number;
}

/// Editable list of pipeline event triggers, mirroring the suggestion-panel styling
const Triggers: React.FC<TriggersProps> = ({ value, onChange, onValidate, mode, showErrors = false, resetKey }) => {
  const [forms, setFormsState] = useState<FormTrigger[]>(() => triggersToForm(value));
  // Re-derive the internal form from value when the parent signals a fresh dataset
  const [prevResetKey, setPrevResetKey] = useState(resetKey);
  if (resetKey !== prevResetKey) {
    setPrevResetKey(resetKey);
    setFormsState(triggersToForm(value));
  }
  useEffect(() => {
    onValidate?.(validateTriggers(forms));
  }, [forms]);
  // update internal state and emit the API-shaped trigger map upward
  const setForms = (next: FormTrigger[]) => {
    setFormsState(next);
    onChange(formToTriggers(next));
  };
  const updateForm = (idx: number, patch: Partial<FormTrigger>) => setForms(forms.map((f, i) => (i === idx ? { ...f, ...patch } : f)));
  const addTrigger = () => setForms([...forms, blankTrigger()]);
  const removeTrigger = (idx: number) => setForms(forms.filter((_, i) => i !== idx));
  if (mode === PipelineFormMode.View) {
    return <TriggerDisplay triggers={value} />;
  }
  const nameErrors = triggerNameErrors(forms);
  return (
    // The section help shows on hovering the triggers content itself (like the description field's
    // `block`-wrapped textarea) rather than from a title/`?`. Lives here so create and edit share it.
    <OverlayTipRight tip={TRIGGERS_TIP} block>
      <Panel>
        {forms.length === 0 && <EmptyNote>No event triggers configured.</EmptyNote>}
        {forms.map((form, idx) => (
          <TriggerCardItem
            key={idx}
            form={form}
            showErrors={showErrors}
            nameError={showErrors ? nameErrors[idx] : null}
            onChange={(patch) => updateForm(idx, patch)}
            onRemove={() => removeTrigger(idx)}
          />
        ))}
        <AddButton type="button" onClick={addTrigger} aria-label="Add trigger">
          <b>+</b>
        </AddButton>
      </Panel>
    </OverlayTipRight>
  );
};

export default Triggers;
