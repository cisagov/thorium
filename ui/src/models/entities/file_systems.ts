import { CreateEntity, Entities, Entity } from './entities';

export type FileSystemMetaFields = {
  sha256: string;
  tools: string[];
};

export type FileSystemCreateMetaFields = {
  sha256: string;
  tools: string[];
};

export type FileSystemMeta = {
  FileSystem: FileSystemMetaFields;
};

export type FileSystemCreateMeta = {
  FileSystem: FileSystemCreateMetaFields;
};

export type FileSystem = Entity & {
  metadata: FileSystemMeta;
};

export type CreateFileSystem = CreateEntity & {
  metadata: FileSystemCreateMeta;
};

export const BlankFileSystem: FileSystem = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.FileSystem,
  metadata: {
    FileSystem: {
      sha256: '',
      tools: [],
    },
  },
  tags: {},
  submitter: '',
  created: '',
};

export const BlankCreateFileSystem: CreateFileSystem = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.FileSystem,
  metadata: {
    FileSystem: {
      sha256: '',
      tools: [],
    },
  },
};
