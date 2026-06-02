import React, { useState, useMemo } from 'react';
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
  ImageFieldsWrapper,
} from './shared.styled';
import { ImageFormMode } from './types';
import AlertBanner from '@components/shared/alerts/AlertBanner';
import FieldBadge from '@components/shared/badges/FieldBadge';
import { OverlayTipRight } from '@components/shared/overlay/tips';

const TOOLTIPS = {
  self: `The resources required to run this image. Running images that exceed their requested resources may be terminated.`,
  cpu: `The number of CPUs that an image will be allowed to consume. Requesting a large amount of CPU may result in an image that can never be scheduled. Units are either whole CPU or integer thousandths of a CPU (mCPU).`,
  memory: `The max amount of memory an image may be allowed to consume. Requesting a large amount of memory may result in an image that can never be scheduled.`,
  ephemeral_storage: `The amount of ephemeral storage that this image requires to run. Requesting a large amount of storage may result in an image that can never be scheduled.`,
  amd_gpu: `The number of AMD GPUs required to run this image. Requesting a large number of GPUs may result in an image that can never be scheduled.`,
  nvidia_gpu: `The number of NVIDIA GPUs required to run this image. Requesting a large number of GPUs may result in an image that can never be scheduled.`,
  burstable_cpu: `The maximum CPU that this image can burst to above its base cpu limits.`,
  burstable_memory: `The maximum memory that this image can burst above its base memory limits.`,
};

const KeyCol = styled.div`
  flex: 0 0 auto;
  min-width: 120px;
`;

const InputRow = styled.div`
  display: flex;
  width: 100%;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
`;

const Input = styled.input`
  flex: 1;
  padding: 6px 12px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  font-size: 14px;
`;

const UnitSelect = styled.select`
  padding: 6px 12px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  font-size: 14px;
  min-width: 80px;
`;

const FieldLabel = styled.em`
  display: block;
`;

enum CpuUnit {
  CPU = 'CPU',
  mCPU = 'mCPU',
}

enum MemoryUnit {
  Gi = 'Gi',
  Mi = 'Mi',
}

interface FormResources {
  cpu: string;
  cpu_units: CpuUnit;
  memory: string;
  memory_units: MemoryUnit;
  ephemeral_storage: string;
  ephemeral_storage_units: MemoryUnit;
  nvidia_gpu: string;
  amd_gpu: string;
  burstable_cpu: string;
  burstable_cpu_units: CpuUnit;
  burstable_memory: string;
  burstable_memory_units: MemoryUnit;
}

export interface ResourcesValue {
  cpu?: number | string;
  memory?: number | string;
  ephemeral_storage?: number | string;
  worker_slots?: number;
  nvidia_gpu?: number;
  amd_gpu?: number;
  burstable?: {
    cpu?: number | string;
    memory?: number | string;
  };
}

interface ResourcesProps {
  value: ResourcesValue;
  onChange: (value: ResourcesValue) => void;
  onValidate?: (hasErrors: boolean) => void;
  mode: ImageFormMode;
  resetKey?: number;
}

function apiToForm(res: ResourcesValue, mode: ImageFormMode): FormResources {
  const isFromApi = mode === ImageFormMode.Edit || mode === ImageFormMode.Copy;

  return {
    cpu: res.cpu != null ? String(res.cpu) : '',
    cpu_units: isFromApi && res.cpu != null ? CpuUnit.mCPU : CpuUnit.CPU,
    memory: res.memory != null ? String(res.memory) : '',
    memory_units: isFromApi && res.memory != null ? MemoryUnit.Mi : MemoryUnit.Gi,
    ephemeral_storage: res.ephemeral_storage != null ? String(res.ephemeral_storage) : '',
    ephemeral_storage_units: isFromApi && res.ephemeral_storage != null ? MemoryUnit.Mi : MemoryUnit.Gi,
    nvidia_gpu: res.nvidia_gpu != null ? String(res.nvidia_gpu) : '',
    amd_gpu: res.amd_gpu != null ? String(res.amd_gpu) : '',
    burstable_cpu: res.burstable?.cpu != null ? String(res.burstable.cpu) : '',
    burstable_cpu_units: isFromApi && res.burstable?.cpu != null ? CpuUnit.mCPU : CpuUnit.CPU,
    burstable_memory: res.burstable?.memory != null ? String(res.burstable.memory) : '',
    burstable_memory_units: isFromApi && res.burstable?.memory != null ? MemoryUnit.Mi : MemoryUnit.Gi,
  };
}

function formToApi(form: FormResources): ResourcesValue {
  const result: ResourcesValue = {};

  if (form.cpu !== '') {
    result.cpu = form.cpu_units === CpuUnit.mCPU ? `${form.cpu}m` : String(form.cpu);
  }
  if (form.memory !== '') {
    result.memory = `${form.memory}${form.memory_units}`;
  }
  if (form.ephemeral_storage !== '') {
    result.ephemeral_storage = `${form.ephemeral_storage}${form.ephemeral_storage_units}`;
  }
  if (form.nvidia_gpu !== '') {
    result.nvidia_gpu = Number(form.nvidia_gpu);
  }
  if (form.amd_gpu !== '') {
    result.amd_gpu = Number(form.amd_gpu);
  }

  const burstable: ResourcesValue['burstable'] = {};
  if (form.burstable_cpu !== '') {
    burstable.cpu = form.burstable_cpu_units === CpuUnit.mCPU ? `${form.burstable_cpu}m` : String(form.burstable_cpu);
  }
  if (form.burstable_memory !== '') {
    burstable.memory = `${form.burstable_memory}${form.burstable_memory_units}`;
  }
  if (Object.keys(burstable).length > 0) {
    result.burstable = burstable;
  }

  return result;
}

function validateResources(form: FormResources): Record<string, string> {
  const errors: Record<string, string> = {};
  if (form.cpu === '' && form.memory !== '') {
    errors.cpu = "cpu can't be empty when memory is specified";
  } else if (form.cpu !== '' && form.cpu_units === CpuUnit.mCPU && Number(form.cpu) < 250) {
    errors.cpu = "cpu can't be less than 250mCPU";
  }
  if (form.memory === '' && form.cpu !== '') {
    errors.memory = "memory can't be empty when cpu is specified";
  } else if (form.memory !== '' && form.memory_units === MemoryUnit.Mi && Number(form.memory) < 500) {
    errors.memory = "memory can't be less than 500Mi";
  }
  return errors;
}

const DisplayResources: React.FC<{ value: ResourcesValue }> = ({ value }) => (
  <>
    <SectionRow>
      <div>
        <OverlayTipRight tip={TOOLTIPS.self}>
          <b>Resources</b>
        </OverlayTipRight>
      </div>
    </SectionRow>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <FieldLabel>cpu: </FieldLabel>
      </KeyCol>
      <ValCol>
        <OverlayTipRight tip={TOOLTIPS.cpu}>
          <FieldBadge field={`${String(parseInt(String(value.cpu ?? 0)))} mCPU`} color="#7e7c7c" />
        </OverlayTipRight>
      </ValCol>
    </SectionRow>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <FieldLabel>memory: </FieldLabel>
      </KeyCol>
      <ValCol>
        <OverlayTipRight tip={TOOLTIPS.memory}>
          <FieldBadge field={`${String(parseInt(String(value.memory ?? 0)))} MiB`} color="#7e7c7c" />
        </OverlayTipRight>
      </ValCol>
    </SectionRow>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <FieldLabel>storage: </FieldLabel>
      </KeyCol>
      <ValCol>
        <OverlayTipRight tip={TOOLTIPS.ephemeral_storage}>
          <FieldBadge field={`${value.ephemeral_storage ?? 0} MiB`} color="#7e7c7c" />
        </OverlayTipRight>
      </ValCol>
    </SectionRow>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <FieldLabel>nvidia gpu: </FieldLabel>
      </KeyCol>
      <ValCol>
        <OverlayTipRight tip={TOOLTIPS.nvidia_gpu}>
          <FieldBadge field={value.nvidia_gpu ?? 0} color="#7e7c7c" />
        </OverlayTipRight>
      </ValCol>
    </SectionRow>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <FieldLabel>amd gpu: </FieldLabel>
      </KeyCol>
      <ValCol>
        <OverlayTipRight tip={TOOLTIPS.amd_gpu}>
          <FieldBadge field={value.amd_gpu ?? 0} color="#7e7c7c" />
        </OverlayTipRight>
      </ValCol>
    </SectionRow>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <FieldLabel>burst cpu: </FieldLabel>
      </KeyCol>
      <ValCol>
        <OverlayTipRight tip={TOOLTIPS.burstable_cpu}>
          <FieldBadge field={`${String(parseInt(String(value.burstable?.cpu ?? 0)))} mCPU`} color="#7e7c7c" />
        </OverlayTipRight>
      </ValCol>
    </SectionRow>
    <SectionRow>
      <IndentCol />
      <KeyCol>
        <FieldLabel>burst memory: </FieldLabel>
      </KeyCol>
      <ValCol>
        <OverlayTipRight tip={TOOLTIPS.burstable_memory}>
          <FieldBadge field={`${String(parseInt(String(value.burstable?.memory ?? 0)))} MiB`} color="#7e7c7c" />
        </OverlayTipRight>
      </ValCol>
    </SectionRow>
  </>
);

const ResourceFields: React.FC<{
  form: FormResources;
  setForm: (f: FormResources) => void;
  errors: Record<string, string>;
}> = ({ form, setForm, errors }) => {
  const update = (key: keyof FormResources, val: string) => {
    setForm({ ...form, [key]: val });
  };

  const numericOnly = (val: string) => val.replace(/[^0-9]/g, '');

  return (
    <ImageFieldsWrapper>
      <FieldLabel>cpu: </FieldLabel>
      <OverlayTipRight tip={TOOLTIPS.cpu}>
        <InputRow>
          <Input
            type="text"
            value={form.cpu}
            placeholder={form.cpu_units === CpuUnit.mCPU ? '1000' : '1'}
            onChange={(e) => update('cpu', numericOnly(e.target.value))}
          />
          <UnitSelect value={form.cpu_units} onChange={(e) => update('cpu_units', e.target.value)}>
            <option value="CPU">CPU</option>
            <option value="mCPU">mCPU</option>
          </UnitSelect>
        </InputRow>
        {errors.cpu && <AlertBanner className="m-2">{errors.cpu}</AlertBanner>}
      </OverlayTipRight>

      <FieldLabel>memory: </FieldLabel>
      <OverlayTipRight tip={TOOLTIPS.memory}>
        <InputRow>
          <Input
            type="text"
            value={form.memory}
            placeholder={form.memory_units === MemoryUnit.Mi ? '2048' : '2'}
            onChange={(e) => update('memory', numericOnly(e.target.value))}
          />
          <UnitSelect value={form.memory_units} onChange={(e) => update('memory_units', e.target.value)}>
            <option value="Gi">GiB</option>
            <option value="Mi">MiB</option>
          </UnitSelect>
        </InputRow>
        {errors.memory && <AlertBanner className="m-2">{errors.memory}</AlertBanner>}
      </OverlayTipRight>

      <FieldLabel>storage: </FieldLabel>
      <OverlayTipRight tip={TOOLTIPS.ephemeral_storage}>
        <InputRow>
          <Input
            type="text"
            value={form.ephemeral_storage}
            placeholder={form.ephemeral_storage_units === MemoryUnit.Mi ? '8192' : '8'}
            onChange={(e) => update('ephemeral_storage', numericOnly(e.target.value))}
          />
          <UnitSelect value={form.ephemeral_storage_units} onChange={(e) => update('ephemeral_storage_units', e.target.value)}>
            <option value="Gi">GiB</option>
            <option value="Mi">MiB</option>
          </UnitSelect>
        </InputRow>
      </OverlayTipRight>

      <FieldLabel>nvidia gpu: </FieldLabel>
      <OverlayTipRight tip={TOOLTIPS.nvidia_gpu}>
        <InputRow>
          <Input
            type="text"
            value={form.nvidia_gpu}
            placeholder="nvidia gpu"
            onChange={(e) => update('nvidia_gpu', numericOnly(e.target.value))}
          />
        </InputRow>
      </OverlayTipRight>

      <FieldLabel>amd gpu: </FieldLabel>
      <OverlayTipRight tip={TOOLTIPS.amd_gpu}>
        <InputRow>
          <Input type="text" value={form.amd_gpu} placeholder="amd gpu" onChange={(e) => update('amd_gpu', numericOnly(e.target.value))} />
        </InputRow>
      </OverlayTipRight>

      <FieldLabel>burst cpu: </FieldLabel>
      <OverlayTipRight tip={TOOLTIPS.burstable_cpu}>
        <InputRow>
          <Input
            type="text"
            value={form.burstable_cpu}
            placeholder={form.burstable_cpu_units === CpuUnit.mCPU ? '2000' : '2'}
            onChange={(e) => update('burstable_cpu', numericOnly(e.target.value))}
          />
          <UnitSelect value={form.burstable_cpu_units} onChange={(e) => update('burstable_cpu_units', e.target.value)}>
            <option value="CPU">CPU</option>
            <option value="mCPU">mCPU</option>
          </UnitSelect>
        </InputRow>
      </OverlayTipRight>

      <FieldLabel>burst mem: </FieldLabel>
      <OverlayTipRight tip={TOOLTIPS.burstable_memory}>
        <InputRow>
          <Input
            type="text"
            value={form.burstable_memory}
            placeholder={form.burstable_memory_units === MemoryUnit.Mi ? '4096' : '4'}
            onChange={(e) => update('burstable_memory', numericOnly(e.target.value))}
          />
          <UnitSelect value={form.burstable_memory_units} onChange={(e) => update('burstable_memory_units', e.target.value)}>
            <option value="Gi">GiB</option>
            <option value="Mi">MiB</option>
          </UnitSelect>
        </InputRow>
      </OverlayTipRight>
    </ImageFieldsWrapper>
  );
};

const Resources: React.FC<ResourcesProps> = ({ value, onChange, onValidate, mode, resetKey }) => {
  const [form, setFormState] = useState<FormResources>(() => apiToForm(value, mode));
  // Re-derive the internal form from value when the parent signals a fresh dataset
  // (e.g. after a save refetch), without clobbering in-progress edits.
  const [prevResetKey, setPrevResetKey] = useState(resetKey);
  if (resetKey !== prevResetKey) {
    setPrevResetKey(resetKey);
    setFormState(apiToForm(value, mode));
  }
  const errors = useMemo(() => validateResources(form), [form]);

  const setForm = (newForm: FormResources) => {
    setFormState(newForm);
    const validationErrors = validateResources(newForm);
    onValidate?.(Object.keys(validationErrors).length > 0);
    onChange(formToApi(newForm));
  };

  if (mode === ImageFormMode.View) {
    return <DisplayResources value={value} />;
  }

  const isEdit = mode === ImageFormMode.Edit;

  if (isEdit) {
    return (
      <SectionRow>
        <EditSpacer />
        <EditMiddle>
          <OverlayTipRight tip={TOOLTIPS.self}>
            <b>Resources</b>
          </OverlayTipRight>
        </EditMiddle>
        <EditFieldCol>
          <ResourceFields form={form} setForm={setForm} errors={errors} />
        </EditFieldCol>
      </SectionRow>
    );
  }

  return (
    <SectionRow>
      <TitleCol>
        <h5>Resources</h5>
      </TitleCol>
      <FieldCol>
        <ResourceFields form={form} setForm={setForm} errors={errors} />
      </FieldCol>
    </SectionRow>
  );
};

export default Resources;
