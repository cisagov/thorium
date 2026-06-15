import React, { useCallback, useRef } from 'react';
import styled from 'styled-components';

// project imports
import { TabItem, TabsProps } from './types';
import { firstEnabledTabIndex, lastEnabledTabIndex, nextEnabledTabIndex } from './step';
import { OverlayTipTop } from '@components/shared/overlay/tips';

// spec: ./Tabs.spec.md

const TabList = styled.div<{ $flush: boolean }>`
  display: flex;
  align-items: stretch;
  gap: 2px;
  border-bottom: ${({ $flush }) => ($flush ? 'none' : '1px solid var(--thorium-panel-border)')};
  flex-wrap: wrap;
`;

const TabButton = styled.button<{ $active: boolean }>`
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border: none;
  background: ${({ $active }) => ($active ? 'var(--thorium-highlight-panel-bg)' : 'transparent')};
  /* the active label uses text-safe --thorium-selected-text (highlight-text is an accent tone that
     fails AA on the highlight panel in Dark/Ocean/Crab); the accent is reserved for the indicator bar */
  color: ${({ $active }) => ($active ? 'var(--thorium-selected-text)' : 'var(--thorium-secondary-text)')};
  font-weight: ${({ $active }) => ($active ? 600 : 500)};
  font-size: 0.95rem;
  cursor: pointer;
  white-space: nowrap;
  border-top-left-radius: 6px;
  border-top-right-radius: 6px;
  transition:
    background 0.15s ease,
    color 0.15s ease;

  /* active indicator bar — uses the theme accent so it matches selected nav styling */
  &::after {
    content: '';
    position: absolute;
    left: 8px;
    right: 8px;
    bottom: -1px;
    height: 2px;
    border-radius: 2px 2px 0 0;
    background: ${({ $active }) => ($active ? 'var(--thorium-highlight-text)' : 'transparent')};
    transition: background 0.15s ease;
  }

  &:hover:not(:disabled) {
    background: var(--thorium-highlight-panel-bg);
    color: var(--thorium-text);
  }

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px var(--thorium-link-text);
    border-radius: 6px;
  }

  &:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
`;

// an action icon rendered inside the active tab button (a span, not a nested <button>, which is invalid
// HTML) — it stops click propagation so activating it fires the action without re-selecting the tab
const ActionIcon = styled.span<{ $disabled: boolean }>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-left: 2px;
  padding: 2px;
  border-radius: 4px;
  color: inherit;
  cursor: ${({ $disabled }) => ($disabled ? 'not-allowed' : 'pointer')};
  opacity: ${({ $disabled }) => ($disabled ? 0.5 : 1)};

  &:hover {
    background: ${({ $disabled }) => ($disabled ? 'transparent' : 'var(--thorium-highlight-panel-bg)')};
  }
`;

const CountBadge = styled.span<{ $active: boolean }>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 20px;
  height: 20px;
  padding: 0 6px;
  border-radius: 10px;
  font-size: 0.75rem;
  font-weight: 600;
  line-height: 1;
  /* both states pair a panel background with --thorium-text so the count reads at AA in all four
     themes (the previous accent-on-accent / secondary-on-highlight pairs fell below the threshold) */
  background: ${({ $active }) => ($active ? 'var(--thorium-highlight-panel-bg)' : 'var(--thorium-secondary-panel-bg)')};
  color: var(--thorium-text);
  border: 1px solid var(--thorium-panel-border);
`;

/**
 * Controlled, themeable, accessible tab bar.
 *
 * Presentation-only: the consumer owns the active key and renders the corresponding panel.
 * Supports an optional count badge per tab and full keyboard navigation (arrow keys, Home/End)
 * over the enabled tabs via a roving tabindex.
 */
function Tabs<K extends string = string>({ tabs, active, onChange, className, 'aria-label': ariaLabel, flush = false }: TabsProps<K>) {
  const buttonRefs = useRef<(HTMLButtonElement | null)[]>([]);

  // move selection/focus to the next enabled tab in the given direction
  const focusTab = useCallback(
    (index: number) => {
      const tab = tabs[index];
      if (!tab || tab.disabled) return;
      onChange(tab.key);
      buttonRefs.current[index]?.focus();
    },
    [tabs, onChange],
  );

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent, currentIndex: number) => {
      switch (event.key) {
        case 'ArrowRight':
        case 'ArrowDown':
          event.preventDefault();
          focusTab(nextEnabledTabIndex(tabs, currentIndex, 1));
          break;
        case 'ArrowLeft':
        case 'ArrowUp':
          event.preventDefault();
          focusTab(nextEnabledTabIndex(tabs, currentIndex, -1));
          break;
        case 'Home':
          event.preventDefault();
          focusTab(firstEnabledTabIndex(tabs));
          break;
        case 'End':
          event.preventDefault();
          focusTab(lastEnabledTabIndex(tabs));
          break;
        default:
          break;
      }
    },
    [tabs, focusTab],
  );

  return (
    <TabList role="tablist" aria-label={ariaLabel} className={className} $flush={flush}>
      {tabs.map((tab: TabItem<K>, index) => {
        const isActive = tab.key === active;
        // the action shows only on the active tab; while it's shown its tip replaces the tab's own tip
        const showAction = isActive && tab.action != null;
        const action = tab.action;
        const button = (
          <TabButton
            ref={(el) => {
              buttonRefs.current[index] = el;
            }}
            type="button"
            role="tab"
            id={`tab-${tab.key}`}
            aria-selected={isActive}
            aria-controls={`tabpanel-${tab.key}`}
            tabIndex={isActive ? 0 : -1}
            disabled={tab.disabled}
            $active={isActive}
            onClick={() => !tab.disabled && onChange(tab.key)}
            onKeyDown={(e) => handleKeyDown(e, index)}
          >
            <span>{tab.label}</span>
            {tab.count != null && (
              <CountBadge $active={isActive}>
                {tab.count}
                {tab.countMore ? '+' : ''}
              </CountBadge>
            )}
            {showAction && action && (
              <OverlayTipTop tip={action.tip}>
                <ActionIcon
                  role="button"
                  aria-label={action.ariaLabel}
                  aria-disabled={action.disabled || undefined}
                  tabIndex={action.disabled ? -1 : 0}
                  $disabled={!!action.disabled}
                  // stop propagation so activating the icon runs the action without re-selecting the tab
                  onClick={(e) => {
                    e.stopPropagation();
                    if (!action.disabled) action.onClick();
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      e.stopPropagation();
                      if (!action.disabled) action.onClick();
                    }
                  }}
                >
                  {action.icon}
                </ActionIcon>
              </OverlayTipTop>
            )}
          </TabButton>
        );
        // when the action is shown its tip replaces the tab's own tip so the two don't both appear
        return tab.tip && !showAction ? (
          <OverlayTipTop key={tab.key} tip={tab.tip}>
            {button}
          </OverlayTipTop>
        ) : (
          <React.Fragment key={tab.key}>{button}</React.Fragment>
        );
      })}
    </TabList>
  );
}

export default Tabs;
