import ReactDatePicker from 'react-datepicker';

// project imports
import { safeStringToDateConversion } from '@utilities/inputs';

interface DatePickerProps {
  max?: string | Date | null | undefined;
  min?: string | Date | null | undefined;
  selected: string | Date | null | undefined;
  disabled: boolean;
  onChange: (date: Date | null) => void;
}

/**
 * A `react-datepicker` wrapper that accepts either ISO date strings or `Date` values and safely
 * coerces them to `Date` before rendering, clearing invalid/unparseable values.
 *
 * @param max - Latest selectable date (string or Date); ignored if unparseable.
 * @param min - Earliest selectable date (string or Date); ignored if unparseable.
 * @param selected - The currently selected date (string or Date), or null/undefined.
 * @param disabled - Whether the picker is disabled.
 * @param onChange - Called with the selected `Date`, or `null` when cleared.
 */
const DatePicker: React.FC<DatePickerProps> = ({ max = null, min = null, selected = null, disabled, onChange }) => {
  let safeMax: Date | undefined = undefined;
  let safeMin: Date | undefined = undefined;
  let safeSelected: Date | undefined = undefined;
  if (max && typeof max == 'string') {
    const maxDate = safeStringToDateConversion(max);
    if (maxDate) {
      safeMax = maxDate;
    }
  } else if (max && max instanceof Date) {
    safeMax = max;
  }
  if (min && typeof min == 'string') {
    const minDate = safeStringToDateConversion(min);
    if (minDate) {
      safeMin = minDate;
    }
  } else if (min && min instanceof Date) {
    safeMin = min;
  }
  if (selected && typeof selected == 'string') {
    const selectedDate = safeStringToDateConversion(selected);
    if (selectedDate) {
      safeSelected = selectedDate;
    }
  } else if (selected && selected instanceof Date) {
    safeSelected = selected;
  }
  return (
    <ReactDatePicker
      isClearable={true}
      maxDate={safeMax}
      minDate={safeMin}
      selected={safeSelected}
      disabled={disabled}
      onChange={(date) => onChange(date instanceof Date ? date : null)}
    />
  );
};

export default DatePicker;
