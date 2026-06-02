import { ElasticIndex } from '@models/search';
import { DropdownOption, EditSession, OmnibarEditMode } from './EditingTypes';
import { Clause, ClauseCondition, ClauseDraft, ClauseIsMulti, GetConditionHelpText, GetValueString } from './ClauseTypes';
import { multiCategories, OmnibarOptionMap } from './options';
import { getTagColorClass } from '@components/tags';

function filterOptionsOnText(options: DropdownOption[], text: string): DropdownOption[] {
  if (!text)
    return options.sort((a, b) => {
      //if both numbers sort on that, otherwise string sort
      const aVal = a.value.trim();
      const bVal = b.value.trim();
      const aNum = Number(aVal);
      const bNum = Number(bVal);
      if (!isNaN(aNum) && !isNaN(bNum)) {
        return aNum - bNum;
      }
      return aVal.localeCompare(bVal);
    });

  const normalizedText = text.toLowerCase().trim();
  const matches = options
    .filter((option) => {
      const normalizedOpt = option.value.toLowerCase().trim();
      return normalizedOpt.includes(normalizedText);
    })
    .sort((a, b) => a.value.localeCompare(b.value));

  //add 'text' option if it's an option and not present in typed matches
  const textEntry = options.find((obj) => obj.category === 'text');
  if (textEntry && !matches.some((obj) => obj.category === 'text')) {
    matches.push(textEntry);
  }
  return matches;
}

export function getDropdownOptions(
  clauses: Clause[],
  editState: EditSession,
  defaults: OmnibarOptionMap,
  currClause: ClauseDraft,
): DropdownOption[] {
  if (editState.mode == OmnibarEditMode.Idle) return [];

  const exisitingCategories = new Set(clauses.map((c) => c.category));

  //field: Just return top-level default options
  if (editState.part == 'category') {
    const opts = Object.keys(defaults)
      .filter((category) => {
        return !exisitingCategories.has(category) || multiCategories.has(category);
      })
      .map((category) => {
        const categoryObj = defaults[category];
        return {
          category: category,
          value: category,
          helpText: categoryObj.helpText,
        };
      });
    return filterOptionsOnText(opts, editState.textDraft);
  }

  //part is not category
  const category = editState.clauseDraft.category;
  if (category == undefined) {
    return [];
  }
  const categoryObj = defaults[category];

  if (editState.part == 'field') {
    const opts = Object.keys(categoryObj.fields).map((field) => {
      return {
        category: category,
        value: field,
      };
    });
    return filterOptionsOnText(opts, editState.textDraft);
  }
  //field should be defined at this point. get defaults for comparison and
  //value based on the field
  let field = currClause.field;
  if (field === undefined) {
    field = '';
  }

  const fieldObject = categoryObj.fields[field];

  if (editState.part == 'condition') {
    if (fieldObject !== undefined) {
      return fieldObject.conditions.map((cond) => {
        return {
          value: cond,
          category: '',
          helpText: GetConditionHelpText(cond),
        };
      });
    }
    return []; //shouldn't reach this, give no options
  }

  //only value left.
  const checkedValues = currClause.values ? currClause.values : [];
  if (fieldObject !== undefined) {
    const opts = fieldObject.values.map((value) => {
      if (checkedValues.includes(value)) {
        return { category: '', value: value, hasCheckmark: true };
      }
      return { category: '', value: value };
    });
    return filterOptionsOnText(opts, editState.textDraft);
  }
  return [];
}

export function getSearchTextFromClauses(clauses: Clause[]): string {
  let query = '';
  const textSearch = clauses.filter((clause) => clause.field == 'text');
  if (textSearch.length > 0) {
    query = textSearch
      .map((clause) => {
        if (ClauseIsMulti(clause)) {
          return clause.value.values.join(' ');
        }
        return clause.value.value;
      })
      .join(' ');
  }
  return query;
}

export function getGroupsFromClauses(clauses: Clause[]): string[] {
  let groups: string[] = [];
  const groupClauses = clauses.filter((clause) => clause.field == 'group');
  groupClauses.forEach((clause) => {
    if (ClauseIsMulti(clause)) {
      groups = clause.value.values;
    } else {
      groups = [clause.value.value];
    }
  });
  return groups;
}

export function getIndexesFromClauses(clauses: Clause[]): ElasticIndex[] {
  let indexes: ElasticIndex[] = [];
  const indexClauses = clauses.filter((clause) => clause.field == 'index');
  indexClauses.forEach((clause) => {
    if (ClauseIsMulti(clause)) {
      indexes = clause.value.values.map((value) => value as ElasticIndex);
    } else {
      indexes = [clause.value.value as ElasticIndex];
    }
  });
  return indexes;
}

export function getLimitFromClauses(clauses: Clause[], defaultLimit: number = 25): number {
  let limit = defaultLimit;
  const limitClauses = clauses.filter((clause) => clause.field == 'limit');
  limitClauses.forEach((clause) => {
    if (!ClauseIsMulti(clause)) {
      if (Number.isInteger(Number(clause.value.value))) {
        limit = Number.parseInt(clause.value.value);
      }
    }
  });
  return limit;
}

export function getStringFieldFromClauses(clauses: Clause[], field: string): string {
  let result = '';
  const filteredClauses = clauses.filter((clause) => clause.field == field);
  filteredClauses.forEach((clause) => {
    if (!ClauseIsMulti(clause)) {
      result = clause.value.value;
    }
  });
  return result;
}

export function getHiddenTagsFromClauses(clauses: Clause[]): string[] {
  let result: string[] = [];
  const hiddenTagClauses = clauses.filter((clause) => clause.field == 'hidden tags');
  hiddenTagClauses.forEach((clause) => {
    if (ClauseIsMulti(clause)) {
      result = Array.from(new Set([...result, ...clause.value.values]));
    }
  });
  return result;
}

/**
 * Test whether a string `value` satisfies every clause targeting `field`, honoring each clause's
 * condition: `includes` is a (case-sensitive) substring match, `is` (or any other single condition)
 * is an exact match, and a multi-value condition (`is one of`) matches when `value` exactly equals
 * one of the listed values. Returns `true` when no clause targets the field (i.e. no constraint).
 *
 * @param clauses - All active omnibar clauses.
 * @param field - The clause field to match against (e.g. `'username'`, `'name'`, `'creator'`).
 * @param value - The candidate value from the item being filtered.
 * @returns `true` if the value satisfies all clauses for the field.
 */
export function matchesStringClauses(clauses: Clause[], field: string, value: string): boolean {
  return clauses
    .filter((clause) => clause.field == field)
    .every((clause) => {
      if (ClauseIsMulti(clause)) {
        return clause.value.values.includes(value);
      }
      if (clause.condition === ClauseCondition.Includes) {
        return value.includes(clause.value.value);
      }
      return value === clause.value.value;
    });
}

export function getStringFieldListFromClauses(clauses: Clause[], field: string): string[] {
  const result: string[] = [];
  const filteredClauses = clauses.filter((clause) => clause.field == field);
  filteredClauses.forEach((clause) => {
    if (ClauseIsMulti(clause)) {
      result.push(...clause.value.values);
    } else {
      result.push(clause.value.value);
    }
  });
  return result;
}

export function getClauseColorClass(clause: Clause): string {
  if (clause.category === 'tag') {
    return getTagColorClass(clause.field, GetValueString(clause));
  }
  //TOOD: add tag-specific
  return 'basic-clause';
}
