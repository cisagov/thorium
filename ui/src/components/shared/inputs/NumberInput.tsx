import React, { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';

const Wrapper = styled.div<{ $disabled: boolean }>`
  display: flex;
  align-items: center;
  border-radius: 0.375rem;
  background-color: var(--thorium-secondary-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  overflow: hidden;

  &:focus-within {
    background-color: var(--thorium-highlight-panel-bg);
    box-shadow:
      inset 0 0 1px var(--thorium-panel-border),
      0 0 6px var(--thorium-highlight-panel-border);
  }

  ${(p) =>
    p.$disabled &&
    `
    filter: brightness(60%);
    cursor: not-allowed;
  `}
`;

const Input = styled.input`
  width: 100%;
  padding: 0.375rem 0.75rem;
  font-size: inherit;
  font-family: inherit;
  line-height: 1.5;
  background: transparent;
  color: var(--thorium-text);
  border: none;
  outline: none;
  -moz-appearance: textfield;

  &::-webkit-inner-spin-button,
  &::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }

  &:disabled {
    cursor: not-allowed;
  }

  &::placeholder {
    color: var(--thorium-secondary-text);
    opacity: 1;
  }

  &:focus::placeholder,
  &:hover::placeholder {
    color: transparent;
  }
`;

const SpinButtons = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1px;
  flex-shrink: 0;
  padding: 0.15rem 0.25rem;
`;

const SpinButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 13px;
  padding: 0;
  line-height: 1;
  font-size: 0.5rem;
  border: none;
  border-radius: 2px;
  background: none;
  color: var(--thorium-secondary-text);
  cursor: pointer;

  &:hover {
    color: var(--thorium-text);
  }

  &:active {
    color: var(--thorium-highlight-text);
  }

  &:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }
`;

export interface NumberInputProps {
  value: number | null;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  required?: boolean;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

const NumberInput: React.FC<NumberInputProps> = ({
  value,
  onChange,
  min,
  max,
  step = 1,
  required = true,
  disabled = false,
  placeholder,
  className,
}) => {
  const [display, setDisplay] = useState(value !== null ? String(value) : '');
  const inputRef = useRef<HTMLInputElement>(null);
  const isFocused = useRef(false);

  useEffect(() => {
    if (!isFocused.current) setDisplay(value !== null ? String(value) : '');
  }, [value]);

  const clamp = (n: number) => {
    let v = n;
    if (min !== undefined && v < min) v = min;
    if (max !== undefined && v > max) v = max;
    return v;
  };

  const commit = (n: number) => {
    const clamped = clamp(n);
    onChange(clamped);
    setDisplay(String(clamped));
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setDisplay(raw);
    if (raw === '' || raw === '-') {
      if (!required) onChange(null);
    } else {
      const parsed = parseFloat(raw);
      if (!isNaN(parsed)) onChange(clamp(parsed));
    }
  };

  const handleBlur = () => {
    isFocused.current = false;
    if (display === '' || display === '-') {
      if (required) {
        commit(min ?? 0);
      } else {
        onChange(null);
        setDisplay('');
      }
    } else {
      const parsed = parseFloat(display);
      commit(isNaN(parsed) ? (min ?? 0) : parsed);
    }
  };

  const handleFocus = (e: React.FocusEvent<HTMLInputElement>) => {
    isFocused.current = true;
    e.target.select();
  };

  const numericValue = value ?? min ?? 0;
  const increment = () => commit(numericValue + step);
  const decrement = () => commit(numericValue - step);

  return (
    <Wrapper className={className} $disabled={disabled}>
      <Input
        ref={inputRef}
        type="number"
        value={display}
        onChange={handleChange}
        onFocus={handleFocus}
        onBlur={handleBlur}
        disabled={disabled}
        placeholder={placeholder}
        min={min}
        max={max}
        step={step}
      />
      <SpinButtons>
        <SpinButton onClick={increment} disabled={disabled || (max !== undefined && numericValue >= max)} aria-label="Increase">
          &#9650;
        </SpinButton>
        <SpinButton onClick={decrement} disabled={disabled || (min !== undefined && numericValue <= min)} aria-label="Decrease">
          &#9660;
        </SpinButton>
      </SpinButtons>
    </Wrapper>
  );
};

export default NumberInput;
