// project imports
import { Clause } from './ClauseTypes';
import { clausesCodec, OmniState, timeCodec } from './urlState';
import { TimeSelection } from './timepicker/utils';
import { useUrlState } from '@utilities/url/useUrlState';

export type OmnibarUrlState = {
  clauses: Clause[];
  setClauses: (next: Clause[]) => void;
  time: TimeSelection;
  setTime: (next: TimeSelection) => void;
};

/**
 * URL-backed omnibar state: a drop-in replacement for the `useState` pairs that browse pages use
 * to hold their omnibar `clauses` and `time`. Clauses and time are bound through two independent
 * codecs (disjoint URL keys), so the omnibar component itself stays unchanged — only the owner of
 * the lifted state changes from local state to the URL.
 *
 * Pages without a time picker (Images/Pipelines/Users/Groups) simply ignore `time`/`setTime`;
 * an `all` selection serializes to nothing.
 */
export function useOmnibarUrlState(defaults: OmniState): OmnibarUrlState {
  const [clauses, setClauses] = useUrlState(clausesCodec(defaults.clauses), defaults.clauses);
  const [time, setTime] = useUrlState(timeCodec(defaults.time), defaults.time);
  return { clauses, setClauses, time, setTime };
}
