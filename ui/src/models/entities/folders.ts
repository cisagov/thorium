import { CreateEntity, Entities, Entity } from './entities';

export type FolderMetaFields = {
  names_sha256: string;
  data_sha256: string;
  all_sha256: string;
};

export type FolderCreateMetaFields = {
  names_sha256: string;
  data_sha256: string;
  all_sha256: string;
};

export type FolderMeta = {
  Folder: FolderMetaFields;
};

export type FolderCreateMeta = {
  Folder: FolderCreateMetaFields;
};

export type Folder = Entity & {
  metadata: FolderMeta;
};

export type CreateFolder = CreateEntity & {
  metadata: FolderCreateMeta;
};

export const BlankFolder: Folder = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.Folder,
  metadata: {
    Folder: {
      names_sha256: '',
      data_sha256: '',
      all_sha256: '',
    },
  },
  tags: {},
  submitter: '',
  created: '',
};

export const BlankCreateFolder: CreateFolder = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.Folder,
  metadata: {
    Folder: {
      names_sha256: '',
      data_sha256: '',
      all_sha256: '',
    },
  },
};
