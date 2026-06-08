import { Clause } from '@components/pages/search/omnibar/ClauseTypes';
import { TimeSelection, TimeSelectionToStrings } from '@components/pages/search/omnibar/timepicker/utils';
import { getGroupsFromClauses, getHiddenTagsFromClauses, getLimitFromClauses } from '@components/pages/search/omnibar/utils';
import { Filters } from '@models/search';

export function OmniClauseAndTimeToFilter(clauses: Clause[], time: TimeSelection, defaultLimit: number = 25): Filters {
  const groups = getGroupsFromClauses(clauses);
  const limit = getLimitFromClauses(clauses, defaultLimit);

  const hidden_tags = getHiddenTagsFromClauses(clauses);

  const [end, start] = TimeSelectionToStrings(time);
  return {
    limit: limit,
    groups: groups,
    start: start,
    end: end,
    hideTags: hidden_tags,
  };
}
