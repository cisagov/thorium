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
import SelectableArray from '@components/shared/inputs/selectable/SelectableArray';
import SelectableDictionary from '@components/shared/inputs/selectable/SelectableDictionary';
import ToggleSwitch from '@components/shared/inputs/ToggleSwitch';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import type {
  Dependencies as DependenciesType,
  SampleDependencySettings,
  RepoDependencySettings,
  ResultDependencySettings,
  EphemeralDependencySettings,
  TagDependencySettings,
  KwargDependencyValue,
} from '@models/images';
import { DependencyPassStrategy } from '@models/images';

const TOOLTIPS = {
  self: `The dependencies an image needs to run. These might include files, repos, results from other tools, or ephemeral files that passed in as reaction arguments.`,
  samples: {
    location: `Destination path to download file(s) into when running this image.`,
    kwarg: `Argument used to pass this image the path to sample dependencies: --some-arg.`,
    strategy: `The method used to pass in dependencies as arguments: file path, file names, directory path, or disabled.`,
  },
  repos: {
    location: `Destination path to download repo(s) into when running this image.`,
    kwarg: `Argument for passing this image the path to dependencies: --some-arg.`,
    strategy: `The method used to pass in dependencies as arguments: file path, directory path, or disabled.`,
  },
  results: {
    location: `Destination path to download result(s) from other tools into when running this image.`,
    kwarg: `Arguments used to pass the result dependency paths to the image.`,
    strategy: `The method used to pass in dependencies as arguments: file path, directory path, or disabled.`,
    names: `A list of result file names that this image depends on from the dependent images. Default behavior is to pull all results from the images.`,
    images: `A list of Thorium images to pull results from for this image. You must select a group for the image you are creating to see the available images for the result dependency.`,
  },
  ephemeral: {
    location: `Destination path to download ephemeral files into when running this image.`,
    kwarg: `Argument for passing this image the path to ephemeral file dependencies: --some-arg.`,
    strategy: `The method used to pass in dependencies as arguments: file path, directory path, or disabled.`,
    names: `A list of ephemeral file names that this image depends on. Ephemeral files are passed into the reaction and are purged after a reaction is completed.`,
  },
  tags: {
    enabled: `Whether tags for target samples will be downloaded before this image is run.`,
    kwarg: `Argument for passing this image the path to the JSON formatted tags file: --some-arg.`,
    location: `Destination path to download JSON tags file to when running this image.`,
    strategy: `The method used to pass in tags dependencies as arguments: file path, directory path, or disabled.`,
  },
};

const DEPENDENCY_STRATEGIES: DependencyPassStrategy[] = [
  DependencyPassStrategy.Paths,
  DependencyPassStrategy.Names,
  DependencyPassStrategy.Directory,
  DependencyPassStrategy.Disabled,
];

enum ResultsKwargType {
  List = 'List',
  Map = 'Map',
  None = 'None',
}

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

const Select = styled.select`
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

const SectionTitle = styled.h6`
  font-weight: 600;
  margin-bottom: 8px;
`;

const Divider = styled.hr`
  border-color: var(--thorium-panel-border);
  margin: 12px 0;
`;

const SwitchLabel = styled.span`
  font-size: 13px;
  font-weight: 600;
  color: var(--thorium-secondary-text);
`;

interface FormDependencies {
  samples: {
    location: string;
    kwarg: string;
    strategy: DependencyPassStrategy;
  };
  repos: {
    location: string;
    kwarg: string;
    strategy: DependencyPassStrategy;
  };
  results: {
    location: string;
    kwarg_type: ResultsKwargType;
    kwarg_list: string;
    kwarg_map: { key: string; value: string }[];
    strategy: DependencyPassStrategy;
    names: string[];
    images: string[];
  };
  ephemeral: {
    location: string;
    kwarg: string;
    strategy: DependencyPassStrategy;
    names: string[];
  };
  tags: {
    enabled: boolean;
    location: string;
    kwarg: string;
    strategy: DependencyPassStrategy;
  };
}

interface DependenciesProps {
  value: DependenciesType;
  onChange: (value: DependenciesType) => void;
  onValidate?: (hasErrors: boolean) => void;
  images: string[];
  mode: ImageFormMode;
  disabled?: boolean;
  resetKey?: number;
}

function resolveKwargType(kwarg?: KwargDependencyValue): ResultsKwargType {
  if (!kwarg || kwarg === 'None') return ResultsKwargType.None;
  if (typeof kwarg === 'object' && 'List' in kwarg) return ResultsKwargType.List;
  if (typeof kwarg === 'object' && 'Map' in kwarg) return ResultsKwargType.Map;
  return ResultsKwargType.None;
}

function apiToForm(deps: DependenciesType): FormDependencies {
  const kwargType = resolveKwargType(deps.results?.kwarg);

  let kwargListVal = '';
  if (kwargType === ResultsKwargType.List && typeof deps.results?.kwarg === 'object' && 'List' in deps.results.kwarg) {
    kwargListVal = deps.results.kwarg.List;
  }

  let kwargMapEntries: { key: string; value: string }[] = [{ key: '', value: '' }];
  if (kwargType === ResultsKwargType.Map && typeof deps.results?.kwarg === 'object' && 'Map' in deps.results.kwarg) {
    const mapObj = deps.results.kwarg.Map;
    const entries = Object.entries(mapObj).map(([k, v]) => ({ key: k, value: v }));
    if (entries.length > 0) {
      kwargMapEntries = entries;
    }
  }

  const resultsImages = deps.results?.images ?? [];
  const resultsNames = deps.results?.names ?? [];

  return {
    samples: {
      location: deps.samples?.location ?? '',
      kwarg: deps.samples?.kwarg ?? '',
      strategy: deps.samples?.strategy ?? DependencyPassStrategy.Paths,
    },
    repos: {
      location: deps.repos?.location ?? '',
      kwarg: deps.repos?.kwarg ?? '',
      strategy: deps.repos?.strategy ?? DependencyPassStrategy.Paths,
    },
    results: {
      location: deps.results?.location ?? '',
      kwarg_type: kwargType,
      kwarg_list: kwargListVal,
      kwarg_map: kwargMapEntries,
      strategy: deps.results?.strategy ?? DependencyPassStrategy.Paths,
      names: resultsNames.length > 0 ? resultsNames : [],
      images: resultsImages.length > 0 ? resultsImages : [],
    },
    ephemeral: {
      location: deps.ephemeral?.location ?? '',
      kwarg: deps.ephemeral?.kwarg ?? '',
      strategy: deps.ephemeral?.strategy ?? DependencyPassStrategy.Paths,
      names: deps.ephemeral?.names ?? [],
    },
    tags: {
      enabled: deps.tags?.enabled ?? false,
      location: deps.tags?.location ?? '',
      kwarg: deps.tags?.kwarg ?? '',
      strategy: deps.tags?.strategy ?? DependencyPassStrategy.Paths,
    },
  };
}

function formToApi(form: FormDependencies): DependenciesType {
  const result: DependenciesType = {};

  // Samples
  const samples: SampleDependencySettings = { strategy: form.samples.strategy };
  if (form.samples.location) samples.location = form.samples.location;
  if (form.samples.kwarg) samples.kwarg = form.samples.kwarg;
  result.samples = samples;

  // Repos
  const repos: RepoDependencySettings = { strategy: form.repos.strategy };
  if (form.repos.location) repos.location = form.repos.location;
  if (form.repos.kwarg) repos.kwarg = form.repos.kwarg;
  result.repos = repos;

  // Results
  const results: ResultDependencySettings = { strategy: form.results.strategy };
  if (form.results.location) results.location = form.results.location;

  if (form.results.kwarg_type === ResultsKwargType.List && form.results.kwarg_list) {
    results.kwarg = { List: form.results.kwarg_list };
  } else if (form.results.kwarg_type === ResultsKwargType.Map) {
    const mapObj: Record<string, string> = {};
    for (const entry of form.results.kwarg_map) {
      if (entry.key.trim()) {
        mapObj[entry.key.trim()] = entry.value;
      }
    }
    if (Object.keys(mapObj).length > 0) {
      results.kwarg = { Map: mapObj };
    }
  } else {
    results.kwarg = 'None';
  }

  const filteredResultNames = form.results.names.filter((n) => n !== '');
  if (filteredResultNames.length > 0) results.names = filteredResultNames;

  const filteredResultImages = form.results.images.filter((n) => n !== '');
  if (filteredResultImages.length > 0) results.images = filteredResultImages;

  result.results = results;

  // Ephemeral
  const ephemeral: EphemeralDependencySettings = { strategy: form.ephemeral.strategy };
  if (form.ephemeral.location) ephemeral.location = form.ephemeral.location;
  if (form.ephemeral.kwarg) ephemeral.kwarg = form.ephemeral.kwarg;
  const filteredEphNames = form.ephemeral.names.filter((n) => n !== '');
  if (filteredEphNames.length > 0) ephemeral.names = filteredEphNames;
  result.ephemeral = ephemeral;

  // Tags
  const tags: TagDependencySettings = { enabled: form.tags.enabled };
  if (form.tags.enabled) {
    if (form.tags.location) tags.location = form.tags.location;
    if (form.tags.kwarg) tags.kwarg = form.tags.kwarg;
    tags.strategy = form.tags.strategy;
  }
  result.tags = tags;

  return result;
}

// View mode display
const DisplayDependencies: React.FC<{ value: DependenciesType }> = ({ value }) => {
  if (!value || Object.keys(value).length === 0) return null;

  const renderField = (label: string, field: string | boolean | string[] | null | undefined) => (
    <SectionRow>
      <SubIndent />
      <SubKeyCol>
        <em>{`${label}: `}</em>
      </SubKeyCol>
      <SubValCol>
        <FieldBadge field={field} color="#7e7c7c" />
      </SubValCol>
    </SectionRow>
  );

  const renderKwargDisplay = () => {
    const kwarg = value.results?.kwarg;
    if (typeof kwarg === 'object' && kwarg && 'Map' in kwarg) {
      return <FieldBadge field={kwarg.Map} color="#7e7c7c" />;
    }
    return <FieldBadge field={kwarg} color="#7e7c7c" />;
  };

  return (
    <>
      {/* Samples */}
      <SectionRow>
        <IndentCol />
        <KeyCol>
          <em>samples:</em>
        </KeyCol>
        <ValCol />
      </SectionRow>
      {renderField('location', value.samples?.location)}
      {renderField('kwarg', value.samples?.kwarg)}
      {renderField('strategy', value.samples?.strategy)}

      {/* Repos */}
      <SectionRow>
        <IndentCol />
        <KeyCol>
          <em>repos:</em>
        </KeyCol>
        <ValCol />
      </SectionRow>
      {renderField('location', value.repos?.location)}
      {renderField('kwarg', value.repos?.kwarg)}
      {renderField('strategy', value.repos?.strategy)}

      {/* Results */}
      <SectionRow>
        <IndentCol />
        <KeyCol>
          <em>results:</em>
        </KeyCol>
        <ValCol />
      </SectionRow>
      {renderField('images', value.results?.images)}
      {value.results?.images && value.results.images.length > 0 && (
        <>
          {renderField('location', value.results?.location)}
          <SectionRow>
            <SubIndent />
            <SubKeyCol>
              <em>kwarg:</em>
            </SubKeyCol>
            <SubValCol>{renderKwargDisplay()}</SubValCol>
          </SectionRow>
          {renderField('strategy', value.results?.strategy)}
          {renderField('names', value.results?.names)}
        </>
      )}

      {/* Ephemeral */}
      <SectionRow>
        <IndentCol />
        <KeyCol>
          <em>ephemeral:</em>
        </KeyCol>
        <ValCol />
      </SectionRow>
      {renderField('location', value.ephemeral?.location)}
      {renderField('kwarg', value.ephemeral?.kwarg)}
      {renderField('strategy', value.ephemeral?.strategy)}
      {renderField('names', value.ephemeral?.names)}

      {/* Tags */}
      <SectionRow>
        <IndentCol />
        <KeyCol>
          <em>tags:</em>
        </KeyCol>
        <ValCol />
      </SectionRow>
      {renderField('enabled', value.tags?.enabled)}
      {renderField('location', value.tags?.location)}
      {renderField('kwarg', value.tags?.kwarg)}
      {renderField('strategy', value.tags?.strategy)}
    </>
  );
};

// Editable form inputs
const DependencyInputs: React.FC<{
  form: FormDependencies;
  setForm: (f: FormDependencies) => void;
  availableImages: string[];
  disabled: boolean;
}> = ({ form, setForm, availableImages, disabled }) => {
  const updateSection = <K extends keyof FormDependencies>(
    section: K,
    key: keyof FormDependencies[K],
    val: FormDependencies[K][keyof FormDependencies[K]],
  ) => {
    setForm({
      ...form,
      [section]: { ...form[section], [key]: val },
    });
  };

  return (
    <ImageFieldsWrapper>
      {/* Samples */}
      <SectionTitle>Samples</SectionTitle>
      <FieldGroup>
        <Label>Location</Label>
        <OverlayTipRight tip={TOOLTIPS.samples.location}>
          <Input
            type="text"
            value={form.samples.location}
            placeholder="/tmp/thorium/samples"
            disabled={disabled}
            onChange={(e) => updateSection('samples', 'location', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>kwarg</Label>
        <OverlayTipRight tip={TOOLTIPS.samples.kwarg}>
          <Input
            type="text"
            value={form.samples.kwarg}
            placeholder="--input-file"
            disabled={disabled}
            onChange={(e) => updateSection('samples', 'kwarg', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>Strategy</Label>
        <OverlayTipRight tip={TOOLTIPS.samples.strategy}>
          <Select
            value={form.samples.strategy}
            disabled={disabled}
            onChange={(e) => updateSection('samples', 'strategy', e.target.value as DependencyPassStrategy)}
          >
            {DEPENDENCY_STRATEGIES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
        </OverlayTipRight>
      </FieldGroup>

      <Divider />

      {/* Repos */}
      <SectionTitle>Repos</SectionTitle>
      <FieldGroup>
        <Label>Location</Label>
        <OverlayTipRight tip={TOOLTIPS.repos.location}>
          <Input
            type="text"
            value={form.repos.location}
            placeholder="/tmp/thorium/repos"
            disabled={disabled}
            onChange={(e) => updateSection('repos', 'location', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>kwarg</Label>
        <OverlayTipRight tip={TOOLTIPS.repos.kwarg}>
          <Input
            type="text"
            value={form.repos.kwarg}
            placeholder="--input-repo"
            disabled={disabled}
            onChange={(e) => updateSection('repos', 'kwarg', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>Strategy</Label>
        <OverlayTipRight tip={TOOLTIPS.repos.strategy}>
          <Select
            value={form.repos.strategy}
            disabled={disabled}
            onChange={(e) => updateSection('repos', 'strategy', e.target.value as DependencyPassStrategy)}
          >
            {DEPENDENCY_STRATEGIES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
        </OverlayTipRight>
      </FieldGroup>

      <Divider />

      {/* Results */}
      <SectionTitle>Results</SectionTitle>
      <FieldGroup>
        <Label>Location</Label>
        <OverlayTipRight tip={TOOLTIPS.results.location}>
          <Input
            type="text"
            value={form.results.location}
            placeholder="/tmp/thorium/prior-results"
            disabled={disabled}
            onChange={(e) => updateSection('results', 'location', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>kwarg</Label>
        <OverlayTipRight tip={TOOLTIPS.results.kwarg}>
          <Select
            value={form.results.kwarg_type}
            disabled={disabled}
            onChange={(e) => updateSection('results', 'kwarg_type', e.target.value as ResultsKwargType)}
            style={{ marginBottom: '8px' }}
          >
            {Object.values(ResultsKwargType).map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </Select>
          {form.results.kwarg_type === ResultsKwargType.Map && (
            <SelectableDictionary
              entries={form.results.kwarg_map}
              disabled={disabled}
              setEntries={(entries: { key: string; value: string }[]) => updateSection('results', 'kwarg_map', entries)}
              keyPlaceholder="New Variable"
              valuePlaceholder="New Value"
              trim={true}
              keys={availableImages}
              deleted={undefined}
              setDeleted={undefined}
            />
          )}
          {form.results.kwarg_type === ResultsKwargType.List && (
            <Input
              type="text"
              value={form.results.kwarg_list}
              placeholder="--input-result"
              disabled={disabled}
              onChange={(e) => updateSection('results', 'kwarg_list', e.target.value.trim())}
            />
          )}
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>Strategy</Label>
        <OverlayTipRight tip={TOOLTIPS.results.strategy}>
          <Select
            value={form.results.strategy}
            disabled={disabled}
            onChange={(e) => updateSection('results', 'strategy', e.target.value as DependencyPassStrategy)}
          >
            {DEPENDENCY_STRATEGIES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>File Names</Label>
        <OverlayTipRight tip={TOOLTIPS.results.names}>
          <SelectableArray
            initialEntries={form.results.names.length > 0 ? form.results.names : ['']}
            setEntries={(names: string[]) => updateSection('results', 'names', names)}
            disabled={disabled}
            placeholder="Filename"
            trim={false}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>Images</Label>
        <OverlayTipRight tip={TOOLTIPS.results.images}>
          <SelectableArray
            initialEntries={form.results.images.length > 0 ? form.results.images : ['']}
            setEntries={(images: string[]) => updateSection('results', 'images', images)}
            disabled={disabled || availableImages.length === 0}
            placeholder={availableImages}
            trim={false}
          />
        </OverlayTipRight>
      </FieldGroup>

      <Divider />

      {/* Ephemeral */}
      <SectionTitle>Ephemeral</SectionTitle>
      <FieldGroup>
        <Label>Location</Label>
        <OverlayTipRight tip={TOOLTIPS.ephemeral.location}>
          <Input
            type="text"
            value={form.ephemeral.location}
            placeholder="/tmp/thorium/ephemeral"
            disabled={disabled}
            onChange={(e) => updateSection('ephemeral', 'location', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>kwarg</Label>
        <OverlayTipRight tip={TOOLTIPS.ephemeral.kwarg}>
          <Input
            type="text"
            value={form.ephemeral.kwarg}
            placeholder="--ephemeral-file"
            disabled={disabled}
            onChange={(e) => updateSection('ephemeral', 'kwarg', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>Strategy</Label>
        <OverlayTipRight tip={TOOLTIPS.ephemeral.strategy}>
          <Select
            value={form.ephemeral.strategy}
            disabled={disabled}
            onChange={(e) => updateSection('ephemeral', 'strategy', e.target.value as DependencyPassStrategy)}
          >
            {DEPENDENCY_STRATEGIES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>File Names</Label>
        <OverlayTipRight tip={TOOLTIPS.ephemeral.names}>
          <SelectableArray
            initialEntries={form.ephemeral.names.length > 0 ? form.ephemeral.names : ['']}
            setEntries={(names: string[]) => updateSection('ephemeral', 'names', names)}
            disabled={disabled}
            placeholder="Filename"
            trim={false}
          />
        </OverlayTipRight>
      </FieldGroup>

      <Divider />

      {/* Tags */}
      <SectionTitle>Tags</SectionTitle>
      <SwitchRow>
        <SwitchLabel>Enabled</SwitchLabel>
        <OverlayTipRight tip={TOOLTIPS.tags.enabled}>
          <ToggleSwitch
            checked={form.tags.enabled}
            disabled={disabled}
            onChange={() => updateSection('tags', 'enabled', !form.tags.enabled)}
          />
        </OverlayTipRight>
      </SwitchRow>
      <FieldGroup>
        <Label>Location</Label>
        <OverlayTipRight tip={TOOLTIPS.tags.location}>
          <Input
            type="text"
            value={form.tags.location}
            placeholder="/tmp/thorium/prior-tags"
            disabled={disabled}
            onChange={(e) => updateSection('tags', 'location', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>kwarg</Label>
        <OverlayTipRight tip={TOOLTIPS.tags.kwarg}>
          <Input
            type="text"
            value={form.tags.kwarg}
            placeholder="--prior-tags"
            disabled={disabled}
            onChange={(e) => updateSection('tags', 'kwarg', e.target.value.trim())}
          />
        </OverlayTipRight>
      </FieldGroup>
      <FieldGroup>
        <Label>Strategy</Label>
        <OverlayTipRight tip={TOOLTIPS.tags.strategy}>
          <Select
            value={form.tags.strategy}
            disabled={disabled}
            onChange={(e) => updateSection('tags', 'strategy', e.target.value as DependencyPassStrategy)}
          >
            {DEPENDENCY_STRATEGIES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
        </OverlayTipRight>
      </FieldGroup>
    </ImageFieldsWrapper>
  );
};

const Dependencies: React.FC<DependenciesProps> = ({ value, onChange, images, mode, disabled = false, resetKey }) => {
  const [form, setFormState] = useState<FormDependencies>(() => apiToForm(value ?? {}));
  // Re-derive the internal form from value when the parent signals a fresh dataset
  // (e.g. after a save refetch), without clobbering in-progress edits.
  const [prevResetKey, setPrevResetKey] = useState(resetKey);
  if (resetKey !== prevResetKey) {
    setPrevResetKey(resetKey);
    setFormState(apiToForm(value ?? {}));
  }

  const setForm = (newForm: FormDependencies) => {
    setFormState(newForm);
    onChange(formToApi(newForm));
  };

  if (mode === ImageFormMode.View) {
    return (
      <>
        <SectionRow>
          <div>
            <OverlayTipRight tip={TOOLTIPS.self}>
              <b>Dependencies</b>
            </OverlayTipRight>
          </div>
        </SectionRow>
        <DisplayDependencies value={value} />
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
            <b>Dependencies</b>
          </OverlayTipRight>
        </EditMiddle>
        <EditFieldCol>
          <DependencyInputs form={form} setForm={setForm} availableImages={images} disabled={disabled} />
        </EditFieldCol>
      </SectionRow>
    );
  }

  return (
    <SectionRow>
      <TitleCol>
        <h5>Dependencies</h5>
      </TitleCol>
      <FieldCol>
        <DependencyInputs form={form} setForm={setForm} availableImages={images} disabled={disabled} />
      </FieldCol>
    </SectionRow>
  );
};

export default Dependencies;
