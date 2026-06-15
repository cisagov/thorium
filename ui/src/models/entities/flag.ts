// project imports
import { CreateEntity, Entities, Entity } from './entities';

/// How confident we are in a flag.
export enum Confidence {
  /// This is known to be a fact
  Fact = 'Fact',
  /// This is more than likely true
  Likely = 'Likely',
  /// This may or may not be true (50/50 odds)
  Unsure = 'Unsure',
  /// This is unlikely to be true and should be validated
  Untrusted = 'Untrusted',
}

/// A flag is a reason that something is interesting, odd, or suspicious.
export type FlagMetaFields = {
  /// How suspicious this flag is where higher numbers are more suspicious
  suspicion: number;
  /// How confident/reliable this flag is
  confidence: Confidence;
  /// The interesting, odd, or suspicious characteristic
  content: string | null;
  /// The reason for this flag
  reasoning: string;
};

/// The create-time metadata shape for a flag; identical to {@link FlagMetaFields}.
export type FlagCreateMetaFields = FlagMetaFields;

export type FlagMeta = {
  Flag: FlagMetaFields;
};

export type FlagCreateMeta = {
  Flag: FlagCreateMetaFields;
};

export type Flag = Entity<Entities.Flag>;

export type CreateFlag = CreateEntity<Entities.Flag>;

export const BlankFlag: Flag = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.Flag,
  metadata: {
    Flag: {
      suspicion: 0,
      confidence: Confidence.Unsure,
      content: null,
      reasoning: '',
    },
  },
  tags: {},
  submitter: '',
  created: '',
  image: null,
};

export const BlankCreateFlag: CreateFlag = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.Flag,
  metadata: {
    Flag: {
      suspicion: 0,
      confidence: Confidence.Unsure,
      content: null,
      reasoning: '',
    },
  },
};
