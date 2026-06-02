import React, { useState } from 'react';
import styled from 'styled-components';
import { FaTrash } from 'react-icons/fa';

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
  IMAGE_FIELDS_MAX_WIDTH,
} from './shared.styled';
import { ImageFormMode } from './types';
import AlertBanner from '@components/shared/alerts/AlertBanner';
import FieldBadge from '@components/shared/badges/FieldBadge';
import ToggleSwitch from '@components/shared/inputs/ToggleSwitch';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import type { Volume, ConfigMap, Secret } from '@models/volumes';
import { VolumeTypes, HostPathTypes } from '@models/volumes';

const TOOLTIPS = {
  self: `Kubernetes volumes to map in during image run. Volumes can be host paths or configuration files.`,
  name: `The name of this volume. Names must be unique and can only contain alphanumeric characters and dashes.`,
  archetype: `The volume type. Volume types determine mount behavior and the volume source.`,
  mode: `The linux file permissions for mounted volumes (octal).`,
  optional: `Whether this volume is required to run the image.`,
  host_path: `Path on host to mount into the image.`,
  host_path_type: `The type of the source target being mounted.`,
  nfs_server: `The NFS server hostname to mount the NFS share from.`,
  nfs_path: `The NFS share path to mount from the server.`,
  mount_path: `The volume's mount path within the running container image.`,
  sub_path: `The file or directory from the volume to mount.`,
  read_only: `Whether the content in this volume can be modified at runtime. Read only is recommended for most tools.`,
  kustomize: `Whether or not this volume was created manually or through kustomize. Kustomize volumes can only be created by admins.`,
};

const ARCHETYPES = Object.values(VolumeTypes);
const HOST_PATH_TYPES = Object.values(HostPathTypes);

const KeyCol = styled.div`
  flex: 0 0 auto;
  min-width: 140px;
`;

const VolumeRow = styled.div`
  margin: 8px 0;
  display: flex;
  align-items: stretch;
  max-width: ${IMAGE_FIELDS_MAX_WIDTH};
`;

const VolumeFields = styled.div`
  flex: 1;
`;

const DeleteCol = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  max-width: 60px;
  padding-left: 8px;
`;

const FieldRow = styled.div`
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 4px;
`;

const FieldLabel = styled.div`
  min-width: 140px;
  font-size: 13px;
  font-weight: 600;
  color: var(--thorium-secondary-text);
`;

const FieldInput = styled.div`
  flex: 1;
  margin: 4px 0;
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

const AddButton = styled.button`
  padding: 4px 16px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  background: var(--thorium-ok-bg);
  color: var(--thorium-button-text);
  font-weight: 700;
  cursor: pointer;

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const DeleteButton = styled.button`
  padding: 6px 12px;
  border: 1px solid var(--thorium-panel-border);
  border-radius: 4px;
  background: var(--thorium-danger-bg);
  color: var(--thorium-button-text);
  cursor: pointer;
`;

const HR = styled.hr`
  margin: 8px 0;
`;

interface VolumeErrors {
  name?: string;
  archetype?: string;
  mount_path?: string;
  host_path?: { path?: string };
  nfs?: { server?: string; path?: string };
}

interface FormVolume extends Volume {
  errors?: VolumeErrors;
}

interface VolumesProps {
  value: Volume[];
  onChange: (value: Volume[]) => void;
  onValidate?: (hasErrors: boolean) => void;
  mode: ImageFormMode;
  disabled?: boolean;
  resetKey?: number;
}

function blankVolume(): FormVolume {
  return {
    name: '',
    archetype: '' as VolumeTypes,
    mount_path: '',
    sub_path: '',
    read_only: false,
    kustomize: false,
    config_map: { default_mode: undefined, optional: false },
    secret: { default_mode: undefined, optional: false },
    nfs: { path: '', server: '' },
    host_path: { path: '', path_type: undefined },
    errors: { name: 'Required', archetype: 'Required', mount_path: 'Required' },
  };
}

function validateVolumes(vols: FormVolume[]): { volumes: FormVolume[]; hasErrors: boolean } {
  let hasErrors = false;
  const result = vols.map((vol) => {
    const errors: VolumeErrors = {};
    if (!vol.name) {
      errors.name = 'Required';
      hasErrors = true;
    }
    if (!vol.archetype) {
      errors.archetype = 'Required';
      hasErrors = true;
    }
    if (!vol.mount_path) {
      errors.mount_path = 'Required';
      hasErrors = true;
    }

    if (vol.archetype === VolumeTypes.HostPath && vol.host_path && !vol.host_path.path) {
      errors.host_path = { path: 'Required' };
      hasErrors = true;
    }
    if (vol.archetype === VolumeTypes.NFS && vol.nfs) {
      const nfsErrors: { server?: string; path?: string } = {};
      if (!vol.nfs.server) {
        nfsErrors.server = 'Required';
        hasErrors = true;
      }
      if (!vol.nfs.path) {
        nfsErrors.path = 'Required';
        hasErrors = true;
      }
      if (Object.keys(nfsErrors).length) errors.nfs = nfsErrors;
    }

    return { ...vol, errors };
  });
  return { volumes: result, hasErrors };
}

function cleanVolume(vol: FormVolume): Volume {
  const clean: Volume = {
    name: vol.name,
    archetype: vol.archetype,
    mount_path: vol.mount_path,
    // required by the API; default false matches the Rust serde defaults
    read_only: vol.read_only,
    kustomize: vol.kustomize,
  };
  if (vol.sub_path) clean.sub_path = vol.sub_path;

  switch (vol.archetype) {
    case VolumeTypes.ConfigMap:
      clean.config_map = vol.config_map;
      break;
    case VolumeTypes.Secret:
      clean.secret = vol.secret;
      break;
    case VolumeTypes.HostPath:
      clean.host_path = vol.host_path;
      break;
    case VolumeTypes.NFS:
      clean.nfs = vol.nfs;
      break;
  }
  return clean;
}

const DisplayVolumes: React.FC<{ volumes: Volume[] }> = ({ volumes }) => (
  <>
    {volumes.map((vol, idx) => (
      <div key={idx}>
        {Object.entries(vol).map(([key, val]) => (
          <SectionRow key={key}>
            <IndentCol />
            <KeyCol>
              <em>{`${key}: `}</em>
            </KeyCol>
            <ValCol>
              <FieldBadge field={JSON.stringify(val)} color="#7e7c7c" />
            </ValCol>
          </SectionRow>
        ))}
        {idx < volumes.length - 1 && <HR />}
      </div>
    ))}
  </>
);

const VolumeEditor: React.FC<{
  volumes: FormVolume[];
  setVolumes: (vols: FormVolume[]) => void;
  disabled: boolean;
}> = ({ volumes, setVolumes, disabled }) => {
  const updateVolume = (idx: number, key: string, subkey: string, val: unknown) => {
    const copy = structuredClone(volumes);
    if (subkey) {
      (copy[idx] as unknown as Record<string, Record<string, unknown>>)[key][subkey] = val;
    } else {
      (copy[idx] as unknown as Record<string, unknown>)[key] = val;
    }

    if (key === 'archetype') {
      const archetype = val as VolumeTypes;
      if (archetype === VolumeTypes.ConfigMap && !copy[idx].config_map) {
        copy[idx].config_map = { default_mode: undefined, optional: false };
      }
      if (archetype === VolumeTypes.Secret && !copy[idx].secret) {
        copy[idx].secret = { default_mode: undefined, optional: false };
      }
      if (archetype === VolumeTypes.NFS && !copy[idx].nfs) {
        copy[idx].nfs = { path: '', server: '' };
      }
      if (archetype === VolumeTypes.HostPath && !copy[idx].host_path) {
        copy[idx].host_path = { path: '', path_type: undefined };
      }
    }

    setVolumes(copy);
  };

  const addVolume = () => setVolumes([...volumes, blankVolume()]);

  const removeVolume = (idx: number) => setVolumes(volumes.filter((_, i) => i !== idx));

  return (
    <>
      {volumes.map((vol, idx) => (
        <VolumeRow key={idx}>
          <VolumeFields>
            <FieldRow>
              <FieldLabel>Name</FieldLabel>
              <FieldInput>
                <OverlayTipRight tip={TOOLTIPS.name}>
                  <Input placeholder="name" value={vol.name} onChange={(e) => updateVolume(idx, 'name', '', e.target.value)} />
                </OverlayTipRight>
                {vol.errors?.name && <AlertBanner className="m-2">{vol.errors.name}</AlertBanner>}
              </FieldInput>
            </FieldRow>
            <FieldRow>
              <FieldLabel>Archetype</FieldLabel>
              <FieldInput>
                <OverlayTipRight tip={TOOLTIPS.archetype}>
                  <Select value={vol.archetype} onChange={(e) => updateVolume(idx, 'archetype', '', e.target.value)}>
                    {!vol.archetype && <option>Select an Archetype</option>}
                    {ARCHETYPES.map((a) => (
                      <option key={a} value={a}>
                        {a}
                      </option>
                    ))}
                  </Select>
                </OverlayTipRight>
                {vol.errors?.archetype && <AlertBanner className="m-2">{vol.errors.archetype}</AlertBanner>}
              </FieldInput>
            </FieldRow>

            {(vol.archetype === VolumeTypes.ConfigMap || vol.archetype === VolumeTypes.Secret) && (
              <>
                <FieldRow>
                  <FieldLabel>Default Mode (octal)</FieldLabel>
                  <FieldInput>
                    <OverlayTipRight tip={TOOLTIPS.mode}>
                      <Input
                        placeholder="600"
                        value={(vol.archetype === VolumeTypes.ConfigMap ? vol.config_map : vol.secret)?.default_mode ?? ''}
                        onChange={(e) => {
                          const v = e.target.value.replace(/[^0-9]/g, '');
                          updateVolume(
                            idx,
                            vol.archetype === VolumeTypes.ConfigMap ? 'config_map' : 'secret',
                            'default_mode',
                            v === '' ? '' : Number(v),
                          );
                        }}
                      />
                    </OverlayTipRight>
                  </FieldInput>
                </FieldRow>
                <FieldRow>
                  <FieldLabel>Optional</FieldLabel>
                  <FieldInput>
                    <OverlayTipRight tip={TOOLTIPS.optional}>
                      <ToggleSwitch
                        checked={(vol.archetype === VolumeTypes.ConfigMap ? vol.config_map : vol.secret)?.optional ?? false}
                        onChange={() => {
                          const key = vol.archetype === VolumeTypes.ConfigMap ? 'config_map' : 'secret';
                          const current = (vol[key] as ConfigMap | Secret)?.optional ?? false;
                          updateVolume(idx, key, 'optional', !current);
                        }}
                      />
                    </OverlayTipRight>
                  </FieldInput>
                </FieldRow>
              </>
            )}

            {vol.archetype === VolumeTypes.HostPath && (
              <>
                <FieldRow>
                  <FieldLabel>Path</FieldLabel>
                  <FieldInput>
                    <OverlayTipRight tip={TOOLTIPS.host_path}>
                      <Input
                        placeholder="/host/src/path"
                        value={vol.host_path?.path ?? ''}
                        onChange={(e) => updateVolume(idx, 'host_path', 'path', e.target.value)}
                      />
                    </OverlayTipRight>
                    {vol.errors?.host_path?.path && <AlertBanner className="m-2">{vol.errors.host_path.path}</AlertBanner>}
                  </FieldInput>
                </FieldRow>
                <FieldRow>
                  <FieldLabel>Path Type</FieldLabel>
                  <FieldInput>
                    <OverlayTipRight tip={TOOLTIPS.host_path_type}>
                      <Select
                        value={vol.host_path?.path_type ?? ''}
                        onChange={(e) => updateVolume(idx, 'host_path', 'path_type', e.target.value || undefined)}
                      >
                        <option value="">Select a Path Type</option>
                        {HOST_PATH_TYPES.map((pt) => (
                          <option key={pt} value={pt}>
                            {pt}
                          </option>
                        ))}
                      </Select>
                    </OverlayTipRight>
                  </FieldInput>
                </FieldRow>
              </>
            )}

            {vol.archetype === VolumeTypes.NFS && (
              <>
                <FieldRow>
                  <FieldLabel>Server</FieldLabel>
                  <FieldInput>
                    <OverlayTipRight tip={TOOLTIPS.nfs_server}>
                      <Input
                        placeholder="hostname"
                        value={vol.nfs?.server ?? ''}
                        onChange={(e) => updateVolume(idx, 'nfs', 'server', e.target.value)}
                      />
                    </OverlayTipRight>
                    {vol.errors?.nfs?.server && <AlertBanner className="m-2">{vol.errors.nfs.server}</AlertBanner>}
                  </FieldInput>
                </FieldRow>
                <FieldRow>
                  <FieldLabel>Path</FieldLabel>
                  <FieldInput>
                    <OverlayTipRight tip={TOOLTIPS.nfs_path}>
                      <Input
                        placeholder="/path/to/directory"
                        value={vol.nfs?.path ?? ''}
                        onChange={(e) => updateVolume(idx, 'nfs', 'path', e.target.value)}
                      />
                    </OverlayTipRight>
                    {vol.errors?.nfs?.path && <AlertBanner className="m-2">{vol.errors.nfs.path}</AlertBanner>}
                  </FieldInput>
                </FieldRow>
              </>
            )}

            <FieldRow>
              <FieldLabel>Mount Path</FieldLabel>
              <FieldInput>
                <OverlayTipRight tip={TOOLTIPS.mount_path}>
                  <Input
                    placeholder="mount path"
                    value={vol.mount_path}
                    onChange={(e) => updateVolume(idx, 'mount_path', '', e.target.value)}
                  />
                </OverlayTipRight>
                {vol.errors?.mount_path && <AlertBanner className="m-2">{vol.errors.mount_path}</AlertBanner>}
              </FieldInput>
            </FieldRow>
            <FieldRow>
              <FieldLabel>Sub Path</FieldLabel>
              <FieldInput>
                <OverlayTipRight tip={TOOLTIPS.sub_path}>
                  <Input
                    placeholder="sub path"
                    value={vol.sub_path ?? ''}
                    onChange={(e) => updateVolume(idx, 'sub_path', '', e.target.value)}
                  />
                </OverlayTipRight>
              </FieldInput>
            </FieldRow>
            <FieldRow>
              <FieldLabel>Read Only</FieldLabel>
              <FieldInput>
                <OverlayTipRight tip={TOOLTIPS.read_only}>
                  <ToggleSwitch checked={vol.read_only ?? false} onChange={() => updateVolume(idx, 'read_only', '', !vol.read_only)} />
                </OverlayTipRight>
              </FieldInput>
            </FieldRow>
            <FieldRow>
              <FieldLabel>Kustomize</FieldLabel>
              <FieldInput>
                <OverlayTipRight tip={TOOLTIPS.kustomize}>
                  <ToggleSwitch checked={vol.kustomize ?? false} onChange={() => updateVolume(idx, 'kustomize', '', !vol.kustomize)} />
                </OverlayTipRight>
              </FieldInput>
            </FieldRow>
          </VolumeFields>
          <DeleteCol>
            <DeleteButton onClick={() => removeVolume(idx)}>
              <FaTrash />
            </DeleteButton>
          </DeleteCol>
          <HR />
        </VolumeRow>
      ))}
      {!disabled && (
        <AddButton onClick={addVolume} disabled={disabled}>
          <b>+</b>
        </AddButton>
      )}
    </>
  );
};

const Volumes: React.FC<VolumesProps> = ({ value, onChange, onValidate, mode, disabled = false, resetKey }) => {
  const [formVolumes, setFormVolumesState] = useState<FormVolume[]>(() =>
    value.map((v): FormVolume => ({ ...structuredClone(v), errors: {} })),
  );
  // Re-derive the internal form from value when the parent signals a fresh dataset
  // (e.g. after a save refetch), without clobbering in-progress edits.
  const [prevResetKey, setPrevResetKey] = useState(resetKey);
  if (resetKey !== prevResetKey) {
    setPrevResetKey(resetKey);
    setFormVolumesState(value.map((v): FormVolume => ({ ...structuredClone(v), errors: {} })));
  }

  const setFormVolumes = (vols: FormVolume[]) => {
    const { volumes: validated, hasErrors } = validateVolumes(vols);
    setFormVolumesState(validated);
    onValidate?.(hasErrors);
    onChange(validated.map(cleanVolume));
  };

  if (mode === ImageFormMode.View) {
    const hasVolumes = value && value.length > 0;
    return (
      <>
        <SectionRow>
          <KeyCol>
            <OverlayTipRight tip={TOOLTIPS.self}>
              <b>Volumes</b>
            </OverlayTipRight>
          </KeyCol>
          <ValCol>{hasVolumes ? <DisplayVolumes volumes={value} /> : <FieldBadge field="None" color="#7e7c7c" />}</ValCol>
        </SectionRow>
      </>
    );
  }

  if (mode === ImageFormMode.Edit) {
    return (
      <SectionRow>
        <EditSpacer />
        <EditMiddle>
          <OverlayTipRight tip={TOOLTIPS.self}>
            <b>Volumes</b>
          </OverlayTipRight>
        </EditMiddle>
        <EditFieldCol>
          <VolumeEditor volumes={formVolumes} setVolumes={setFormVolumes} disabled={disabled} />
        </EditFieldCol>
      </SectionRow>
    );
  }

  return (
    <SectionRow>
      <TitleCol>
        <h5>Volumes</h5>
      </TitleCol>
      <FieldCol>
        <VolumeEditor volumes={formVolumes} setVolumes={setFormVolumes} disabled={disabled} />
      </FieldCol>
    </SectionRow>
  );
};

export default Volumes;
