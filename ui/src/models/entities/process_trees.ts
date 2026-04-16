import { CreateEntity, Entities, Entity } from './entities';

export type WindowsProcessTreeMetaFields = {
  tools: string[];
};

export type WindowsProcessTreeCreateMetaFields = {
  tools: string[];
};

export type WindowsProcessTreeMeta = {
  WindowsProcessTree: WindowsProcessTreeMetaFields;
};

export type WindowsProcessTreeCreateMeta = {
  WindowsProcessTree: WindowsProcessTreeCreateMetaFields;
};

export type WindowsProcessTree = Entity<Entities.WindowsProcessTree>;

export type CreateWindowsProcessTree = CreateEntity<Entities.WindowsProcessTree>;

export const BlankWindowsProcessTree: WindowsProcessTree = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.WindowsProcessTree,
  metadata: {
    WindowsProcessTree: {
      tools: [],
    },
  },
  tags: {},
  submitter: '',
  created: '',
};

export const BlankCreateWindowsProcessTree: CreateWindowsProcessTree = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.WindowsProcessTree,
  metadata: {
    WindowsProcessTree: {
      tools: [],
    },
  },
};
