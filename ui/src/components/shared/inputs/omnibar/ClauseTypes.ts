import { OmnibarOptionMap } from './options';

/* Package for clause data and all clause related functions.
 *
 * */
export type ClausePart = 'category' | 'field' | 'value' | 'condition';

export type SingleValue = {
  value: string;
};

export type MultiValue = {
  values: string[];
};

//NOTE: add new conditions here
export enum ClauseCondition {
  Is = 'is',
  IsNot = 'is not',
  IsOneOf = 'is one of',
  Are = 'are',
  /// substring match (e.g. "username includes ali" matches "alice")
  Includes = 'includes',
}

//NOTE: add new multi conditions here
const multiConditions = [ClauseCondition.IsOneOf, ClauseCondition.Are] as const;

type MultiCondition = (typeof multiConditions)[number];
type SingleCondition = Exclude<ClauseCondition, MultiCondition>;

type ClauseBase = {
  field: string;
  category: string;
};

export type Clause =
  | (ClauseBase & { condition: SingleCondition; value: SingleValue })
  | (ClauseBase & { condition: MultiCondition; value: MultiValue });

export type ClauseDraft = {
  category?: string;
  field?: string;
  condition?: ClauseCondition;
  value?: string;
  values?: string[];
};

export function parseClauseCondition(input: string): ClauseCondition | undefined {
  const s = input.trim().toLowerCase();
  const values = Object.values(ClauseCondition) as string[];
  if (values.includes(s)) {
    return s as ClauseCondition;
  }
  return undefined;
}

export function ClauseIsMulti(clause: Clause): clause is Extract<Clause, { condition: MultiCondition }> {
  return CondIsMulti(clause.condition);
}

export function CondIsMulti(cond: ClauseCondition): cond is MultiCondition {
  return (multiConditions as readonly ClauseCondition[]).includes(cond);
}

export function AddCategoryToClauseDraft(draft: ClauseDraft, category: string): ClauseDraft {
  return { ...draft, category: category };
}

export function AddFieldToClauseDraft(draft: ClauseDraft, field: string): ClauseDraft {
  return { ...draft, field: field };
}

export function AddConditionToClauseDraft(draft: ClauseDraft, condition: ClauseCondition): ClauseDraft {
  return { ...draft, condition: condition };
}

export function ToggleValueInClauseDraft(draft: ClauseDraft, value: string): ClauseDraft {
  //NOTE: if existing already in array will remove
  if (!draft.condition) {
    throw new Error('Attempting to add a value to a partial clause that does not have a condition set');
  }
  const condition = draft.condition;
  if (CondIsMulti(condition)) {
    const existingValues = draft.values ? draft.values : [];
    if (existingValues.includes(value)) {
      return { ...draft, values: existingValues.filter((val) => val !== value) };
    } else {
      return { ...draft, values: [...existingValues, value] };
    }
  } else {
    return { ...draft, value: value };
  }
}

export function ToggleValueInClause(clause: Clause, value: string): Clause {
  if (ClauseIsMulti(clause)) {
    const exisiting = clause.value.values;
    if (exisiting.includes(value)) {
      return { ...clause, value: { values: exisiting.filter((val) => val !== value) } };
    } else {
      return { ...clause, value: { values: [...clause.value.values, value] } };
    }
  }
  return { ...clause, value: { value: value } };
}

export function DraftIsComplete(draft: ClauseDraft): draft is {
  field: string;
  category: string;
  condition: ClauseCondition;
  value?: string;
  values?: string[];
} {
  return !!draft.category && !!draft.field && !!draft.condition && (!!draft.value || !!draft.values?.length);
}

export function ConvertDraftToClause(draft: ClauseDraft): Clause {
  const field = draft.field;
  const category = draft.category;
  const condition = draft.condition;
  if (!field || !condition || !category) throw new Error('Draft is missing field or condition');
  if (CondIsMulti(condition)) {
    const values = draft.values ?? [];
    if (values.length === 0) throw new Error('Draft is missing values for multi-select condition');
    return { category: category, field: field, condition: condition, value: { values: values } };
  } else {
    const value = draft.value;
    if (!value) throw new Error('draft is missing value');
    return { category: category, field: field, condition: condition, value: { value: value } };
  }
}

export function ConvertClauseToDraft(clause: Clause): ClauseDraft {
  const draft: ClauseDraft = {
    category: clause.category,
    field: clause.field,
    condition: clause.condition,
  };
  if (ClauseIsMulti(clause)) {
    draft.values = clause.value.values;
  } else {
    draft.value = clause.value.value;
  }
  return draft;
}

export function GetValueString(clause: Clause): string {
  if (ClauseIsMulti(clause)) {
    return clause.value.values.join(', ');
  }
  return clause.value.value;
}

export function NewTextClause(textValue: string): Clause {
  return {
    category: 'text',
    field: 'text',
    condition: ClauseCondition.Is,
    value: { value: textValue },
  };
}

export function GetConditionHelpText(cond: ClauseCondition): string {
  switch (cond) {
    case ClauseCondition.Is:
      return 'field matches / has single value';
    case ClauseCondition.IsNot:
      return 'field does not match / have single value';
    case ClauseCondition.IsOneOf:
      return 'field matches / has one or more selected values';
    case ClauseCondition.Are:
      return 'field matches all selected values';
    case ClauseCondition.Includes:
      return 'field contains this value (substring match)';
    default:
      return '';
  }
}

export function GetMostSpecificCondition(text: string): ClauseCondition | undefined {
  const condition_values = Object.values(ClauseCondition)
    .slice()
    .sort((a, b) => b.length - a.length); //sort longest to shortest
  const t = text.toLowerCase();
  return condition_values.find((cond) => t.includes(cond));
}

export type FieldAndCategory = {
  field: string;
  category: string;
};

export function GetValidFields(omniOpts: OmnibarOptionMap): FieldAndCategory[] {
  const fields: FieldAndCategory[] = [];
  Object.keys(omniOpts).forEach((category) => {
    Object.keys(omniOpts[category].fields).forEach((field) => {
      fields.push({ field: field, category: category });
    });
  });
  return fields;
}

export function DefaultClausesEntities(): Clause[] {
  return [
    {
      category: 'hidden tags',
      field: 'hidden tags',
      condition: ClauseCondition.Are,
      value: { values: ['Results', 'Parent', 'submitter'] },
    },
  ];
}
