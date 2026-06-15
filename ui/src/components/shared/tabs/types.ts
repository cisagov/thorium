import React from 'react';

// spec: ./Tabs.spec.md

/**
 * An icon action rendered next to a tab's label while that tab is active (e.g. a "fetch" button).
 * When present on the active tab, its tooltip replaces the tab's own {@link TabItem.tip} so the two
 * tips don't both appear.
 */
export interface TabAction {
  /** Icon content for the action button. */
  icon: React.ReactNode;
  /** Tooltip shown on hover of the action button. */
  tip: string;
  /** Accessible label for the action button. */
  ariaLabel: string;
  /** Invoked when the action button is clicked. */
  onClick: () => void;
  /** Whether the action button is disabled (e.g. while its work is in progress). */
  disabled?: boolean;
}

/** A single tab descriptor for the {@link Tabs} component. */
export interface TabItem<K extends string = string> {
  /** Stable key identifying the tab; passed back to `onChange`. */
  key: K;
  /** Visible label content. */
  label: React.ReactNode;
  /** Optional count rendered as a small badge next to the label. */
  count?: number;
  /** When true, the count badge shows a trailing `+` to signal there are more than {@link count} (e.g. a
   * cursor-paginated list where only the loaded-so-far total is known). */
  countMore?: boolean;
  /** Optional tooltip shown on hover. */
  tip?: string;
  /** Optional icon action shown next to the label while this tab is active. */
  action?: TabAction;
  /** Whether the tab is selectable. */
  disabled?: boolean;
}

/** Props for the controlled {@link Tabs} component. */
export interface TabsProps<K extends string = string> {
  /** The tabs to render, in display order. */
  tabs: TabItem<K>[];
  /** The key of the currently active tab. */
  active: K;
  /** Called with the key of a newly selected tab. */
  onChange: (key: K) => void;
  /** Optional class on the tab list container. */
  className?: string;
  /** Accessible label for the tab list. */
  'aria-label'?: string;
  /** Drop the tab list's own bottom border (e.g. when placed inline on a bordered header). */
  flush?: boolean;
}
