import { CreateEntity, Entities, Entity } from './entities';

export type OtherMetaFields = {};

export type OtherCreateMetaFields = OtherMetaFields;

export type OtherMeta = {
  Other: OtherMetaFields;
};

export type OtherCreateMeta = {
  Other: OtherCreateMetaFields;
};

export type Other = Entity<Entities.Other>;

export type CreateOther = CreateEntity<Entities.Other>;

export const BlankOther: Other = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.Other,
  metadata: {
    Other: {},
  },
  tags: {},
  submitter: '',
  created: '',
};

export const BlankCreateOther: CreateOther = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.Other,
  metadata: {
    Other: {},
  },
};
