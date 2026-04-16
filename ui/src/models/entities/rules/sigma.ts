import { CreateEntity, Entities, Entity } from '../entities';

export enum SigmaRuleAppliesTo {
  /// Apply this rule to windows processes
  WindowsProcesses = 'WindowsProcesses',
  /// Apply this rule to network connections
  NetworkConnections = 'NetworkConnections',
}

/// Automatically promote this sigma rule hit to a flag
export type SigmaAutoFlag = {
  /// The interesting, odd, or suspicious characteristic
  content?: string;
  /// The reason for this Flag
  reasoning: string;
};

export enum SigmaRuleActions {
  Flag = 'Flag',
}

/// Map each action to its payload type
export type SigmaActionActionType = {
  [SigmaRuleActions.Flag]: SigmaAutoFlag;
};

/// A single action variant
export type SigmaActionToTake = {
  [K in keyof SigmaActionActionType]: {
    [P in K]: SigmaActionActionType[K];
  };
}[keyof SigmaActionActionType];

/// The action to take when a sigma rule hits
export type SigmaActionToTake2<T extends keyof SigmaRuleActions> = {
  /// Automatically promote hit to a flag
  [k in T]: SigmaAutoFlag;
};

/// A Sigma rule that can be applied to logs or log like data
export type SigmaRuleset = {
  /// The original unparsed rule
  rule: string;
  /// What types of data this sigma rule applies too
  applies_to: SigmaRuleAppliesTo[];
  /// The score to apply when this rule hits
  score: number;
  /// The actions to take when this sigma rule hits
  actions: SigmaActionToTake[];
};

export type SigmaRuleMetaFields = SigmaRuleset;

export type SigmaRuleMeta = {
  SigmaRule: SigmaRuleset;
};

export type SigmaRuleCreateMeta = {
  SigmaRule: SigmaRuleset;
};

export type SigmaRule = Entity<Entities.SigmaRule>;

export type CreateSigmaRule = CreateEntity<Entities.SigmaRule>;

export const BlankSigmaRule: SigmaRule = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.SigmaRule,
  metadata: {
    SigmaRule: {
      rule: '',
      score: 0,
      applies_to: [],
      actions: [],
    },
  },
  tags: {},
  submitter: '',
  created: '',
};

export const BlankCreateSigmaRule: CreateSigmaRule = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.SigmaRule,
  metadata: {
    SigmaRule: {
      rule: '',
      score: 0,
      applies_to: [],
      actions: [],
    },
  },
};
