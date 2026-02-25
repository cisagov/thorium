// project imports
import { RequestTags } from '../tags';
import { CreateEntity, Entities, Entity } from './entities';

export enum CollectionKind {
  Files = 'Files',
  Repos = 'Repos',
}

export type CollectionMetaFields = {
  collection_kind?: CollectionKind;
  collection_tags?: RequestTags;
  tags_case_insensitive?: boolean;
  ignore_groups?: boolean;
  start?: string | null; // ISO‑8601 datetime string
  end?: string | null; // ISO‑8601 datetime string
};

export type CollectionCreateMetaFields = {
  collection_kind?: CollectionKind;
  collection_tags?: RequestTags;
  tags_case_insensitive?: boolean;
  ignore_groups?: boolean;
  start?: string | null;
  end?: string | null;
};

export type CollectionMeta = {
  Collection: CollectionMetaFields;
};

export type CollectionCreateMeta = {
  Collection: CollectionCreateMetaFields;
};

export type Collection = Entity & {
  metadata: CollectionMeta;
};

export type CreateCollection = CreateEntity & {
  metadata: CollectionCreateMeta;
};

export const BlankCollection: Collection = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.Collection,
  metadata: {
    Collection: {
      collection_kind: CollectionKind.Files,
      collection_tags: {},
      tags_case_insensitive: true,
      ignore_groups: false,
      start: null,
      end: null,
    },
  },
  tags: {},
  submitter: '',
  created: '',
};

export const BlankCreateCollection: CreateCollection = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.Collection,
  metadata: {
    Collection: {
      collection_kind: CollectionKind.Files,
      collection_tags: {},
      tags_case_insensitive: false,
      ignore_groups: false,
      start: null,
      end: null,
    },
  },
};
