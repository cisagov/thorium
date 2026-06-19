import { PresetOptions, RelativeSelection } from './utils';
import { styled } from 'styled-components';

export const PresetOptionList: PresetOptions = {
  'Last 15 minutes': { mode: 'relative', amount: 15, unit: 'minute' },
  'Last 60 minutes': { mode: 'relative', amount: 60, unit: 'minute' },
  'Last 4 hours': { mode: 'relative', amount: 4, unit: 'hour' },
  'Last 24 hours': { mode: 'relative', amount: 24, unit: 'hour' },
  'Last 7 days': { mode: 'relative', amount: 7, unit: 'day' },
  'Last 30 days': { mode: 'relative', amount: 30, unit: 'day' },
  'Last 6 months': { mode: 'relative', amount: 6, unit: 'month' },
  'Last 12 months': { mode: 'relative', amount: 12, unit: 'month' },
  'Last 5 years': { mode: 'relative', amount: 5, unit: 'year' },
  'Last 10 years': { mode: 'relative', amount: 10, unit: 'year' },
};

const PresetContainer = styled.div`
  width: 100%;
  padding: 10px;
`;

const PresetList = styled.ul<{ $rows: number }>`
  list-style: none;
  padding-left: 0;
  margin: 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  grid-template-rows: repeat(${(p) => p.$rows}, auto);
  grid-auto-flow: column;
  gap: 6px 24px;
`;

const PresetLink = styled.button`
  background: none;
  border: 0;
  padding: 0;
  margin: 0;
  font: inherit;
  color: inherit;
  text-decoration: underline;
  cursor: pointer;

  &:hover {
    text-decoration-thickness: 2px;
  }

  &:focus-visible {
    outline: 2px solid currentColor;
    outline-offset: 2px;
  }
`;

type PresetProps = {
  onChange: (next: RelativeSelection) => void;
};

const Presets: React.FC<PresetProps> = ({ onChange }) => {
  return (
    <PresetContainer>
      <PresetList $rows={Math.ceil(Object.keys(PresetOptionList).length / 2)}>
        {Object.entries(PresetOptionList).map(([key, value]) => (
          <li key={key}>
            <PresetLink onClick={() => onChange(value)}>{key}</PresetLink>
          </li>
        ))}
      </PresetList>
    </PresetContainer>
  );
};
export default Presets;
