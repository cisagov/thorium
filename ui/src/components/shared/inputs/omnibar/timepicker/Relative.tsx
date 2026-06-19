import React from 'react';
import { styled } from 'styled-components';
import { RelativeSelection, RelativeUnit } from './utils';
import { Form, FormControl } from 'react-bootstrap';

type Props = {
  time: RelativeSelection;
  min?: number;
  disabled?: boolean;
  idPrefix?: string;
  onChange: (next: RelativeSelection) => void;
};

const UNITS: Array<{ value: RelativeUnit; label: string }> = [
  { value: 'minute', label: 'Minute' },
  { value: 'hour', label: 'Hour' },
  { value: 'day', label: 'Day' },
  { value: 'week', label: 'Week' },
  { value: 'month', label: 'Month' },
  { value: 'year', label: 'Year' },
];

const RelativeContainer = styled.div`
  width: 75%;
  display: inline-flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: space-around;
  gap: 10px;
  padding: 10px;
`;

const RelativeRow = styled.div`
  display: inline-flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-around;
  gap: 10px;
  padding: 10px 0;
`;

const CheckboxLabel = styled.label`
  display: inline-flex;
  align-items: center;
  gap: 8px;
`;

const InlineText = styled.span`
  font-size: 1rem;
  line-height: 1.5;
`;

function clampInt(raw: string, min: number) {
  // accept empty while typing; commit on blur
  const trimmed = raw.trim();
  if (trimmed === '') return { ok: false as const };

  // parse as integer
  const n = Number(trimmed);
  if (!Number.isFinite(n)) return { ok: false as const };

  const int = Math.trunc(n);
  if (int < min) return { ok: true as const, value: min };

  return { ok: true as const, value: int };
}

export default function Relative({ time, min = 1, disabled = false, onChange }: Props) {
  const [draft, setDraft] = React.useState<string>(String(time.amount));

  // If parent value changes (e.g., external reset), sync the draft
  React.useEffect(() => {
    // Only overwrite draft when not actively editing could be added,
    // but this simple sync is often good enough.
    setDraft(String(time.amount));
  }, [time.amount]);

  const commit = React.useCallback(() => {
    const parsed = clampInt(draft, min);
    if (!parsed.ok) {
      setDraft(String(time.amount));
      return;
    }
    if (parsed.value !== time.amount) onChange({ ...time, amount: parsed.value });
    setDraft(String(parsed.value)); // normalize display
  }, [draft, min, onChange, time]);

  const handleValueChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    // Allow only digits (and empty) in the draft; optional but helps.
    const next = e.target.value;
    if (next === '' || /^[0-9]+$/.test(next)) setDraft(next);
  };

  const handleValueBlur: React.FocusEventHandler<HTMLInputElement> = () => {
    commit();
  };

  const handleValueKeyDown: React.KeyboardEventHandler<HTMLInputElement> = (e) => {
    if (e.key === 'Enter') {
      commit();
      (e.target as HTMLInputElement).blur();
    }
    if (e.key === 'Escape') {
      setDraft(String(time.amount));
      (e.target as HTMLInputElement).blur();
    }
  };

  const handleUnitChange: React.ChangeEventHandler<HTMLSelectElement> = (e) => {
    onChange({ ...time, unit: e.target.value as RelativeUnit });
  };

  return (
    <RelativeContainer>
      <RelativeRow>
        <InlineText>Last</InlineText>
        <FormControl
          type="number"
          inputMode="numeric"
          value={draft}
          disabled={disabled}
          onChange={handleValueChange}
          onBlur={handleValueBlur}
          onKeyDown={handleValueKeyDown}
        />
        <Form.Select value={time.unit} disabled={disabled} onChange={handleUnitChange}>
          {UNITS.map((u) => (
            <option key={u.value} value={u.value}>
              {time.amount === 1 ? u.label : u.label + 's'}
            </option>
          ))}
        </Form.Select>
      </RelativeRow>
      <div>
        <CheckboxLabel>
          <input
            id="round_checkbox"
            type="checkbox"
            checked={time.round}
            onChange={(e) => {
              onChange({ ...time, round: e.target.checked });
            }}
          />
          Round to start of day
        </CheckboxLabel>
      </div>
    </RelativeContainer>
  );
}
