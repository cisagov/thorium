import { CreateTags, TagEntry } from 'models/tags';
import { CreateEntity, Entities, Entity, CreateEntityPreprocessor } from './entities';

export enum CollectionKind {
  Files = 'Files',
  Repos = 'Repos',
}

export type CollectionTags = CreateTags;

export type CollectionMetaFields = {
  collection_kind?: CollectionKind;
  collection_tags?: CollectionTags;
  tags_case_insensitive?: boolean;
  ignore_groups?: boolean;
  start?: string | null; // ISO‑8601 datetime string
  end?: string | null; // ISO‑8601 datetime string
};

export type CollectionCreateMetaFields = {
  collection_kind?: CollectionKind;
  collection_tags?: CollectionTags;
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

export class CreateCollectionPreprocessor implements CreateEntityPreprocessor<CreateCollection> {
  editbleTags: TagEntry[] = [];

  preprocess(this: this, collection: CreateCollection): CreateCollection {
    // Convert `[{key, value}]`to CollectionTags map
    const collectionTags: CollectionTags = {};
    this.editbleTags.forEach(({ key, value }) => {
      // skip empty rows
      if (!key) return;
      const trimmedKey = key.trim();
      const trimmedVal = value.trim();
      // ignore rows that have no value
      if (!trimmedVal) return;

      if (!collectionTags[trimmedKey]) collectionTags[trimmedKey] = [];
      // append the value; the API will dedupe values for us
      collectionTags[trimmedKey].push(trimmedVal);
    });
    collection.metadata.Collection.collection_tags = collectionTags;
    console.log(collectionTags);
    return collection;
  }
}

export const BlankCollection: Collection = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.Collection,
  metadata: {
    Collection: {
      collection_kind: CollectionKind.Files, // default to Files
      collection_tags: {},
      tags_case_insensitive: false,
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
