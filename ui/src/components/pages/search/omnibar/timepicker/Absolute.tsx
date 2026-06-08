import DatePicker from 'react-datepicker';
import { AbsoluteSelection } from './utils';
import { styled } from 'styled-components';
import 'react-datepicker/dist/react-datepicker.css';
import { Button } from 'react-bootstrap';

type AbsoluteProps = {
  time: AbsoluteSelection;
  setTime: (next: AbsoluteSelection) => void;
};

const AbsoluteRow = styled.div`
  display: grid;
  grid-template-columns: 70px 1fr 70px;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  padding: 5px 10px;
`;

const Absolute: React.FC<AbsoluteProps> = ({ time, setTime }) => {
  function onChangeStart(next: Date) {
    setTime({ ...time, start: next });
  }

  function onChangeEnd(next: Date) {
    setTime({ ...time, end: next });
  }

  function onClickNowButton() {
    setTime({ ...time, end: new Date() });
  }

  return (
    <>
      <AbsoluteRow>
        <label>Start:</label>
        <DatePicker
          selected={time.start}
          onChange={(d) => onChangeStart(d)}
          timeInputLabel="Time:"
          dateFormat="MM/dd/yyyy h:mm aa"
          showTimeSelect={true}
          isClearable={true}
          timeIntervals={15}
        />
      </AbsoluteRow>
      <AbsoluteRow>
        <label>End:</label>
        <DatePicker
          selected={time.end}
          onChange={(d) => onChangeEnd(d)}
          timeInputLabel="Time:"
          dateFormat="MM/dd/yyyy h:mm aa"
          showTimeSelect
          isClearable
          timeIntervals={15}
        />
        <Button className="primary-btn" onClick={onClickNowButton}>
          Now
        </Button>
      </AbsoluteRow>
    </>
  );
};

export default Absolute;
