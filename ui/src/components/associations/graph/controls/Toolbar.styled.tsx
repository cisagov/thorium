import styled, { css } from 'styled-components';
import { Button, ButtonGroup, Dropdown, Form, Popover, Spinner } from 'react-bootstrap';

export const ToolbarContainer = styled.div`
  position: absolute;
  bottom: 12px;
  left: 12px;
  z-index: 500;
  display: flex;
  align-items: center;
  gap: 6px;
`;

export const ToolbarIconButton = styled.button<{ $active?: boolean }>`
  width: 36px;
  height: 36px;
  border-radius: 8px;
  border: 1px solid var(--thorium-panel-border);
  background: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition:
    background 0.15s,
    border-color 0.15s;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
  }

  ${({ $active }) =>
    $active &&
    css`
      background: var(--thorium-highlight-panel-bg);
      border-color: var(--thorium-highlight-panel-border);
    `}
`;

export const NodeCount = styled.span`
  font-size: 0.75rem;
  color: var(--thorium-secondary-text);
  padding: 0 8px;
  white-space: nowrap;
  user-select: none;
`;

export const StyledPopover = styled(Popover)`
  --bs-popover-bg: var(--thorium-secondary-panel-bg);
  --bs-popover-border-color: var(--thorium-panel-border);
  --bs-popover-body-color: var(--thorium-text);
  --bs-popover-arrow-border: var(--thorium-panel-border);
`;

export const PopoverBody = styled.div`
  display: grid;
  gap: 10px;
  padding: 4px;
  min-width: 200px;
`;

export const ControlRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

export const ControlLabel = styled.label`
  font-size: 0.8rem;
  white-space: nowrap;
  color: var(--thorium-text);
  min-width: 0;
`;

export const RangeInput = styled(Form.Range)`
  flex: 1;
  min-width: 100px;
`;

export const Divider = styled.hr`
  margin: 2px 0;
  border-color: var(--thorium-panel-border);
  opacity: 0.4;
`;

export const FullWidthDropdown = styled(Dropdown).attrs({ as: ButtonGroup })`
  width: 100%;

  .dropdown-toggle {
    width: 100%;
  }
`;

export const FullWidthButton = styled(Button)`
  width: 100%;
`;

export const ButtonRow = styled.div`
  display: flex;
  gap: 6px;
`;

export const FlexButton = styled(Button)`
  flex: 1;
`;

export const ToolbarSpinner = styled(Spinner)`
  width: 12px;
  height: 12px;
  margin-right: 6px;
  border-width: 2px;
`;

export const MenuList = styled.div<{ $inset?: boolean }>`
  display: flex;
  flex-direction: column;
  margin: ${({ $inset }) => ($inset ? '0 -4px' : '-8px -12px')};
  min-width: 120px;
`;

export const MenuItem = styled.button`
  background: none;
  border: none;
  color: var(--thorium-text);
  padding: 8px 16px;
  font-size: 0.82rem;
  text-align: left;
  cursor: pointer;
  position: relative;
  transition: background 0.15s;

  &:hover {
    background: var(--thorium-highlight-panel-bg);
  }

  &:first-child {
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
  }

  &:last-child {
    border-bottom-left-radius: 6px;
    border-bottom-right-radius: 6px;
  }

  &:not(:last-child)::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 16px;
    right: 16px;
    height: 1px;
    background: var(--thorium-panel-border);
    opacity: 0.5;
  }
`;

export const MenuDropdown = styled(Dropdown)`
  width: 100%;
  position: relative;

  .dropdown-toggle {
    background: none;
    border: none;
    color: var(--thorium-text);
    padding: 8px 16px;
    font-size: 0.82rem;
    text-align: left;
    width: 100%;
    cursor: pointer;
    transition: background 0.15s;
    border-radius: 0;

    &:hover,
    &:focus {
      background: var(--thorium-highlight-panel-bg);
      color: var(--thorium-text);
      box-shadow: none;
    }

    &::after {
      float: right;
      margin-top: 6px;
    }
  }

  &:first-child .dropdown-toggle {
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
  }

  &:last-child .dropdown-toggle {
    border-bottom-left-radius: 6px;
    border-bottom-right-radius: 6px;
  }

  &:not(:last-child)::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 16px;
    right: 16px;
    height: 1px;
    background: var(--thorium-panel-border);
    opacity: 0.5;
    pointer-events: none;
  }
`;
