import { Clause } from '@components/shared/inputs/omnibar/ClauseTypes';
import { TimeSelection, TimeSelectionToStrings } from '@components/shared/inputs/omnibar/timepicker/utils';
import {
  getGroupsFromClauses,
  getHiddenTagsFromClauses,
  getLimitFromClauses,
  getTagsFromClauses,
} from '@components/shared/inputs/omnibar/utils';
import { Filters } from '@models/search';

export function OmniClauseAndTimeToFilter(clauses: Clause[], time: TimeSelection, defaultLimit: number = 25): Filters {
  const groups = getGroupsFromClauses(clauses);
  const limit = getLimitFromClauses(clauses, defaultLimit);
  const tags = getTagsFromClauses(clauses);

  const hidden_tags = getHiddenTagsFromClauses(clauses);

  const [end, start] = TimeSelectionToStrings(time);
  return {
    limit: limit,
    groups: groups,
    tags: tags,
    start: start,
    end: end,
    hideTags: hidden_tags,
  };
}
