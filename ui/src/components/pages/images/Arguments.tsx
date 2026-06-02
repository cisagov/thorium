import React, { useState, useMemo, useCallback } from 'react';
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
import AlertBanner from '@components/shared/alerts/AlertBanner';
import FieldBadge from '@components/shared/badges/FieldBadge';
import SelectableArray from '@components/shared/inputs/selectable/SelectableArray';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import type { ImageArgs, ArgStrategyValue } from '@models/images';
import { ArgStrategy } from '@models/images';

const TOOLTIPS = {
  self: `The command line parameters passed to this image when it is run.`,
  entrypoint: `The entrypoint executable or script that the agent will call to run this image. For images run in K8s, leaving this blank will cause the default container entrypoint to be used.`,
  command: `The command arguments to pass the entrypoint of the image.`,
  reaction: `The flag used to pass in the reaction ID of the running image.`,
  repo: `The flag used to pass in a git repo. This is only used by Thorium data generation jobs.`,
  commit: `The flag used to pass in the specific repo commit. This is only used by Thorium data generation jobs.`,
  output: `The flag or arg position used to pass in an output path for this image's results.`,
  kwarg: `The actual flag used to pass in the output path.`,
};

const OUTPUT_TYPES = ['Append', 'Kwarg', 'None'] as const;

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
`;

const Select = styled.select`
  width: 100%;
  padding: 6px 12px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  font-size: 14px;
`;

interface FormArgs {
  entrypoint: string[];
  command: string[];
  reaction: string;
  repo: string;
  commit: string;
  output: string;
  kwarg: string;
}

interface ArgumentsProps {
  value: ImageArgs;
  onChange: (value: ImageArgs) => void;
  onValidate?: (hasErrors: boolean) => void;
  mode: ImageFormMode;
  resetKey?: number;
}

function apiToForm(args: ImageArgs): FormArgs {
  let output = 'None';
  let kwarg = '';
  if (args.output) {
    if (typeof args.output === 'string') {
      output = args.output;
    } else if (typeof args.output === 'object' && 'Kwarg' in args.output) {
      output = 'Kwarg';
      kwarg = args.output.Kwarg;
    }
  }

  return {
    entrypoint: args.entrypoint ?? [],
    command: args.command ?? [],
    reaction: args.reaction ?? '',
    repo: args.repo ?? '',
    commit: args.commit ?? '',
    output,
    kwarg,
  };
}

function formToApi(form: FormArgs): ImageArgs {
  const result: ImageArgs = {};

  const ep = form.entrypoint.filter((s) => s !== '');
  if (ep.length > 0) result.entrypoint = ep;

  const cmd = form.command.filter((s) => s !== '');
  if (cmd.length > 0) result.command = cmd;

  if (form.reaction) result.reaction = form.reaction;
  if (form.repo) result.repo = form.repo;
  if (form.commit) result.commit = form.commit;

  if (form.output === 'Kwarg' && form.kwarg) {
    result.output = { Kwarg: form.kwarg };
  } else if (form.output === 'Append') {
    result.output = ArgStrategy.Append;
  } else {
    result.output = ArgStrategy.None;
  }

  return result;
}

function validateArgs(form: FormArgs): Record<string, string> {
  const errors: Record<string, string> = {};
  if (form.output === 'Kwarg' && !form.kwarg) {
    errors.output = "Kwarg flag must be specified when 'Kwarg' is selected for output";
  }
  return errors;
}

const DisplayArguments: React.FC<{ value: ImageArgs }> = ({ value }) => {
  const fields: [string, string | string[] | ArgStrategyValue | undefined][] = [
    ['entrypoint', value.entrypoint],
    ['command', value.command],
    ['reaction', value.reaction],
    ['repo', value.repo],
    ['commit', value.commit],
    ['output', value.output],
  ];

  return (
    <>
      {fields.map(([key, val]) => (
        <SectionRow key={key}>
          <IndentCol />
          <KeyCol>
            <em>{`${key}: `}</em>
          </KeyCol>
          <ValCol>
            <OverlayTipRight tip={TOOLTIPS[key as keyof typeof TOOLTIPS] ?? ''}>
              <FieldBadge field={val} color="#7e7c7c" />
            </OverlayTipRight>
          </ValCol>
        </SectionRow>
      ))}
    </>
  );
};

const ArgumentFields: React.FC<{
  form: FormArgs;
  setForm: (f: FormArgs) => void;
  errors: Record<string, string>;
}> = ({ form, setForm, errors }) => {
  const update = useCallback(
    (key: keyof FormArgs, val: FormArgs[keyof FormArgs]) => {
      setForm({ ...form, [key]: val });
    },
    [form, setForm],
  );

  return (
    <ImageFieldsWrapper>
      <OverlayTipRight tip={TOOLTIPS.entrypoint}>
        <FieldGroup>
          <Label>Entry Point</Label>
          <SelectableArray
            initialEntries={form.entrypoint.length > 0 ? form.entrypoint.map((s) => s.trim()) : ['']}
            setEntries={(entries: string[]) => update('entrypoint', entries)}
            placeholder="entry point"
            trim={true}
            disabled={false}
          />
        </FieldGroup>
      </OverlayTipRight>

      <OverlayTipRight tip={TOOLTIPS.command}>
        <FieldGroup>
          <Label>Command</Label>
          <SelectableArray
            initialEntries={form.command.length > 0 ? form.command.map((s) => s.trim()) : ['']}
            setEntries={(entries: string[]) => update('command', entries)}
            placeholder="command"
            trim={true}
            disabled={false}
          />
        </FieldGroup>
      </OverlayTipRight>

      <OverlayTipRight tip={TOOLTIPS.reaction}>
        <FieldGroup>
          <Label>Reaction</Label>
          <Input type="text" value={form.reaction} placeholder="reaction" onChange={(e) => update('reaction', e.target.value.trim())} />
        </FieldGroup>
      </OverlayTipRight>

      <OverlayTipRight tip={TOOLTIPS.repo}>
        <FieldGroup>
          <Label>Repo</Label>
          <Input type="text" value={form.repo} placeholder="repo" onChange={(e) => update('repo', e.target.value.trim())} />
        </FieldGroup>
      </OverlayTipRight>

      <OverlayTipRight tip={TOOLTIPS.commit}>
        <FieldGroup>
          <Label>Commit</Label>
          <Input type="text" value={form.commit} placeholder="commit" onChange={(e) => update('commit', e.target.value.trim())} />
        </FieldGroup>
      </OverlayTipRight>

      <OverlayTipRight tip={TOOLTIPS.output}>
        <FieldGroup>
          <Label>Output</Label>
          <Select value={form.output} onChange={(e) => update('output', e.target.value.trim())}>
            {OUTPUT_TYPES.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </Select>
        </FieldGroup>
      </OverlayTipRight>

      {form.output === 'Kwarg' && (
        <OverlayTipRight tip={TOOLTIPS.kwarg}>
          <FieldGroup>
            <Label>Kwarg</Label>
            <Input type="text" value={form.kwarg} placeholder="kwarg option" onChange={(e) => update('kwarg', e.target.value.trim())} />
          </FieldGroup>
        </OverlayTipRight>
      )}

      {errors.output && <AlertBanner className="m-2">{errors.output}</AlertBanner>}
    </ImageFieldsWrapper>
  );
};

const Arguments: React.FC<ArgumentsProps> = ({ value, onChange, onValidate, mode, resetKey }) => {
  const [form, setFormState] = useState<FormArgs>(() => apiToForm(value));
  // Re-derive the internal form from value when the parent signals a fresh dataset
  // (e.g. after a save refetch), without clobbering in-progress edits.
  const [prevResetKey, setPrevResetKey] = useState(resetKey);
  if (resetKey !== prevResetKey) {
    setPrevResetKey(resetKey);
    setFormState(apiToForm(value));
  }
  const errors = useMemo(() => validateArgs(form), [form]);

  const setForm = (newForm: FormArgs) => {
    setFormState(newForm);
    const validationErrors = validateArgs(newForm);
    onValidate?.(Object.keys(validationErrors).length > 0);
    onChange(formToApi(newForm));
  };

  if (mode === ImageFormMode.View) {
    return (
      <>
        <SectionRow>
          <div>
            <OverlayTipRight tip={TOOLTIPS.self}>
              <b>Arguments</b>
            </OverlayTipRight>
          </div>
        </SectionRow>
        <DisplayArguments value={value} />
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
            <b>Arguments</b>
          </OverlayTipRight>
        </EditMiddle>
        <EditFieldCol>
          <ArgumentFields form={form} setForm={setForm} errors={errors} />
        </EditFieldCol>
      </SectionRow>
    );
  }

  return (
    <SectionRow>
      <TitleCol>
        <h5>Arguments</h5>
      </TitleCol>
      <FieldCol>
        <ArgumentFields form={form} setForm={setForm} errors={errors} />
      </FieldCol>
    </SectionRow>
  );
};

export default Arguments;
