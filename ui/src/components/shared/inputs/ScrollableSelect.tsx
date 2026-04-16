import React, { useEffect, useRef, useState } from 'react';
import { Dropdown } from 'react-bootstrap';
import { FaChevronUp, FaChevronDown } from 'react-icons/fa';

interface ScrollableSelectProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  windowSize?: number;
  initialStart?: number;
}

const toggleStyle: React.CSSProperties = {
  width: 60,
  height: 36,
  borderRadius: 8,
  border: '1px solid var(--thorium-panel-border)',
  background: 'var(--thorium-secondary-panel-bg)',
  color: 'var(--thorium-text)',
  fontSize: '0.8rem',
  padding: '4px 8px',
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
};

const menuStyle: React.CSSProperties = {
  minWidth: 60,
  background: 'var(--thorium-secondary-panel-bg)',
  border: '1px solid var(--thorium-panel-border)',
  borderRadius: 8,
  padding: '2px 0',
  // Override Bootstrap .dropdown-item hover which uses --thorium-nav-text (nav bar color, white in Light theme)
  '--bs-dropdown-link-hover-color': 'var(--thorium-text)',
  '--bs-dropdown-link-hover-bg': 'var(--thorium-highlight-panel-bg)',
  '--bs-dropdown-link-active-color': 'var(--thorium-text)',
  '--bs-dropdown-link-active-bg': 'var(--thorium-highlight-panel-bg)',
} as React.CSSProperties;

const itemStyle: React.CSSProperties = {
  color: 'var(--thorium-text)',
  fontSize: '0.8rem',
  padding: '4px 12px',
  cursor: 'pointer',
  textAlign: 'center',
};

const activeItemStyle: React.CSSProperties = {
  ...itemStyle,
  background: 'var(--thorium-highlight-panel-bg)',
  fontWeight: 600,
};

const arrowStyle: React.CSSProperties = {
  ...itemStyle,
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  padding: '4px 12px',
  opacity: 0.7,
};

const ScrollableSelect: React.FC<ScrollableSelectProps> = ({ value, onChange, min = 1, max = Infinity, windowSize = 5, initialStart }) => {
  const [windowStart, setWindowStart] = useState(initialStart ?? min);
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (value < windowStart) {
      setWindowStart(Math.max(min, value));
    } else if (value >= windowStart + windowSize) {
      setWindowStart(value - windowSize + 1);
    }
  }, [value, min, windowSize]);

  const windowEnd = Math.min(windowStart + windowSize - 1, max);
  const canScrollUp = windowStart > min;
  const canScrollDown = windowEnd < max;

  const scrollUp = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setWindowStart((s) => Math.max(min, s - 1));
  };

  const scrollDown = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setWindowStart((s) => (max === Infinity ? s + 1 : Math.min(max - windowSize + 1, s + 1)));
  };

  const items: number[] = [];
  for (let i = windowStart; i <= windowEnd; i++) {
    items.push(i);
  }

  return (
    <Dropdown show={open} onToggle={setOpen}>
      <Dropdown.Toggle as="button" style={toggleStyle} id="scrollable-select-toggle">
        {value}
      </Dropdown.Toggle>
      <Dropdown.Menu ref={menuRef} style={menuStyle} renderOnMount>
        {canScrollUp && (
          <Dropdown.Item style={arrowStyle} onClick={scrollUp}>
            <FaChevronUp size={10} />
          </Dropdown.Item>
        )}
        {items.map((n) => (
          <Dropdown.Item
            key={n}
            active={false}
            style={n === value ? activeItemStyle : itemStyle}
            onClick={() => {
              onChange(n);
              setOpen(false);
            }}
          >
            {n}
          </Dropdown.Item>
        ))}
        {canScrollDown && (
          <Dropdown.Item style={arrowStyle} onClick={scrollDown}>
            <FaChevronDown size={10} />
          </Dropdown.Item>
        )}
      </Dropdown.Menu>
    </Dropdown>
  );
};

export default ScrollableSelect;
