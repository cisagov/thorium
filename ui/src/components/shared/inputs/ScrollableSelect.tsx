import React, { useEffect, useId, useRef, useState } from 'react';
import { Dropdown } from 'react-bootstrap';
import { FaChevronUp, FaChevronDown } from 'react-icons/fa';

// spec: ./ScrollableSelect.spec.md

interface ScrollableSelectProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  windowSize?: number;
  initialStart?: number;
  onOpenChange?: (open: boolean) => void;
  /** When true, the control can't be opened/changed (e.g. while a grow is in flight). */
  disabled?: boolean;
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

const ScrollableSelect: React.FC<ScrollableSelectProps> = ({
  value,
  onChange,
  min = 1,
  max = Infinity,
  windowSize = 5,
  initialStart,
  onOpenChange,
  disabled = false,
}) => {
  const [windowStart, setWindowStart] = useState(initialStart ?? min);
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  // unique per instance so multiple mounted ScrollableSelects don't collide on a shared DOM id
  const toggleId = useId();

  const handleToggle = (nextOpen: boolean) => {
    // don't open while disabled (e.g. a grow is in flight)
    if (disabled) return;
    setOpen(nextOpen);
    onOpenChange?.(nextOpen);
  };

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
    <Dropdown show={open} onToggle={handleToggle}>
      <Dropdown.Toggle
        as="button"
        disabled={disabled}
        style={{ ...toggleStyle, ...(disabled ? { filter: 'var(--thorium-disabled-brightness)', cursor: 'not-allowed' } : {}) }}
        id={toggleId}
      >
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
              // Route close through handleToggle so onOpenChange fires (spec: closing notifies the caller)
              handleToggle(false);
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
