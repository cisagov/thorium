import Tab from 'react-bootstrap/Tab';
import Tabs from 'react-bootstrap/Tabs';
import { useRef, useState } from 'react';
import { styled } from 'styled-components';
import Absolute from './Absolute';
import Relative from './Relative';
import { Overlay, Popover } from 'react-bootstrap';
import Presets from './Presets';
import { AbsoluteSelection, RelativeSelection, startOfLast, TimeSelection } from './utils';
import { FaClock } from 'react-icons/fa';

export type ResolvedTimeRange = { start: Date; end: Date };

const TimePickerContainer = styled.div`
  display: inline-flex;
  align-items: center;
  align-self: center;
  justify-content: center;
  gap: 5px;
  height: 100%;
`;

const TimePickerPopover = styled(Popover)`
  max-width: 500px;
  min-width: 400px;
  border: 1px solid var(--thorium-omnibar-border);
  border-radius: 5px;

  padding: 5px 2px;
  background-color: var(--thorium-panel-bg);
  color: var(--thorium-text);
`;

const TabContent = styled.div`
  margin: 5px;
  padding: 0px 5px;
`;

const TimeButton = styled.button`
  border-radius: 0 5px 5px 0;
  background-color: var(--thorium-omnibar-bg);
  color: var(--thorium-text);
  border: 1px solid var(--thorium-omnibar-border);
  border-left: none;
  height: 100%;

  display: flex;
  justify-content: center;
  align-items: center;
  gap: 5px;
  padding: 0 5px;
`;

type TabKey = 'presets' | 'absolute' | 'relative' | 'all';

type SingleTimePickerProps = {
  time: TimeSelection;
  setTime: (next: TimeSelection) => void;
};

function defaultAbsolute(curr: TimeSelection): AbsoluteSelection {
  if (curr.mode === 'absolute') {
    return curr;
  }
  const start = startOfLast(7, 'day');
  const end = new Date();
  return { mode: 'absolute', start: start, end: end };
}

function defaultRelative(curr: TimeSelection): RelativeSelection {
  if (curr.mode === 'relative') {
    return { mode: 'relative', amount: curr.amount, unit: curr.unit, round: false };
  }
  return { mode: 'relative', amount: 7, unit: 'day', round: false };
}

const SingleTimePicker: React.FC<SingleTimePickerProps> = ({ time, setTime }) => {
  const [absoluteTime, setAbsoluteTime] = useState<AbsoluteSelection>(defaultAbsolute(time));
  const [relativeTime, setRelativeTime] = useState<RelativeSelection>(defaultRelative(time));
  const [activeTab, setActiveTab] = useState<TabKey>(time.mode);
  const target = useRef<HTMLButtonElement | null>(null);
  const [showPopover, setShowPopover] = useState(false);

  const getTextValue = (): string => {
    switch (activeTab) {
      case 'absolute':
        return 'Absolute Range';
      case 'relative':
      case 'presets':
        return `Last ${relativeTime.amount} ${relativeTime.unit}${relativeTime.amount == 1 ? '' : 's'}`;
      case 'all':
        return 'All Time';
    }
  };

  const commit = () => {
    switch (activeTab) {
      case 'absolute':
        setTime(absoluteTime);
        break;
      case 'relative':
      case 'presets':
        setTime(relativeTime);
        break;
      case 'all':
        setTime({ mode: 'all' });
    }
  };

  return (
    <TimePickerContainer>
      <TimeButton ref={target} onClick={() => setShowPopover((s) => !s)} aria-describedby="my-popover" className="text-nowrap">
        <FaClock />
        {getTextValue()}
      </TimeButton>
      <Overlay
        target={target.current}
        show={showPopover}
        placement="bottom"
        rootClose
        onHide={() => {
          commit();
          setShowPopover(false);
        }}
      >
        <TimePickerPopover id="my-popover">
          <Tabs fill activeKey={activeTab} onSelect={(key) => setActiveTab(key as TabKey)}>
            <Tab eventKey="presets" title="Presets">
              <TabContent>
                <Presets onChange={setRelativeTime} />
              </TabContent>
            </Tab>
            <Tab eventKey="absolute" title="Absolute">
              <TabContent>
                <Absolute time={absoluteTime} setTime={setAbsoluteTime} />
              </TabContent>
            </Tab>
            <Tab eventKey="relative" title="Relative">
              <TabContent>
                <Relative time={relativeTime} onChange={setRelativeTime} />
              </TabContent>
            </Tab>
            <Tab eventKey="all" title="All Time">
              <TabContent>
                <p style={{ color: 'inherit' }}>All Time</p>
              </TabContent>
            </Tab>
          </Tabs>
        </TimePickerPopover>
      </Overlay>
    </TimePickerContainer>
  );
};

export default SingleTimePicker;
