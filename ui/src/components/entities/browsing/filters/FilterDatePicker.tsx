import DatePicker from 'react-datepicker';

// project imports
import { safeStringToDateConversion } from '@utilities/inputs';

interface FilterDateProps {
  max?: string | Date | null | undefined;
  min?: string | Date | null | undefined;
  selected: string | Date | null | undefined;
  disabled: boolean;
  onChange: (date: Date | null) => void;
}

const FilterDatePicker: React.FC<FilterDateProps> = ({ max = null, min = null, selected = null, disabled, onChange }) => {
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
    <DatePicker
      //className="date-picker-input"
      isClearable={true}
      maxDate={safeMax}
      minDate={safeMin}
      selected={safeSelected}
      disabled={disabled}
      onChange={(date: any) => onChange(date instanceof Date ? date : null)}
    />
  );
};

export default FilterDatePicker;
