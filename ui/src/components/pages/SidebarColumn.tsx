import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { NavLink, useLocation } from 'react-router-dom';
import { FaChevronDown, FaChevronRight } from 'react-icons/fa';
import styled, { createGlobalStyle, css, keyframes } from 'styled-components';

// project imports
import { NAV_ITEMS, NavCategory, NavSubItem } from './navConfig';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import { RequireAuth, useAuth } from '@utilities/auth';
import { RoleKey, UserInfo } from '@models/users';
import { CanvasMargin, scaling } from '@styles';

// spec: ./SPEC.md
// spec: ./Page.spec.md

const ICON_SIZE = 22;
const FLYOUT_DELAY_MS = 200;
const FLYOUT_GAP = 4;

// --- Styled components ---

const NavPanel = styled.div`
  z-index: 0;
  left: 0;
  top: ${CanvasMargin.top}px;
  padding: 0.25rem 0.5rem 0.5rem;
  position: fixed;
  height: calc(100% - ${CanvasMargin.top}px);
  border-right: 0.05px groove var(--thorium-panel-border);
  color: var(--thorium-nav-text);
  background-color: var(--thorium-nav-panel-bg);
  overflow-y: auto;
  width: 170px;

  @media (max-width: ${scaling.xl}) {
    width: 60px;
  }

  @media (max-width: ${scaling.sm}) {
    display: none;
  }
`;

const SideCol = styled.div`
  flex: 1 !important;
  flex-basis: 170px !important;
  flex-shrink: 0 !important;
  flex-grow: 0 !important;

  @media (max-width: ${scaling.xl}) {
    flex-basis: 60px !important;
  }

  @media (max-width: ${scaling.sm}) {
    flex: 0 !important;
    display: none;
  }
`;

const CategoryContainer = styled.div`
  position: relative;
  padding-bottom: 0.35rem;
  margin-bottom: 0.35rem;
  border-bottom: 1px solid var(--thorium-panel-border);

  &:last-child {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
  }
`;

const hoverStyle = css`
  color: var(--thorium-highlight-text);
  background-color: var(--thorium-highlight-panel-bg);
`;

const CategoryRow = styled.div<{ $active?: boolean }>`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.5rem;
  cursor: pointer;
  border-radius: 6px;
  color: ${(p) => (p.$active ? 'var(--thorium-highlight-text)' : 'var(--thorium-nav-text)')};
  text-decoration: none;
  transition:
    background-color 0.15s ease,
    color 0.15s ease;

  @media (max-width: ${scaling.xl}) {
    justify-content: center;
  }

  &:hover {
    ${hoverStyle}
  }
`;

const CategoryLabel = styled.span`
  white-space: nowrap;
  font-size: 0.9rem;
  line-height: ${ICON_SIZE}px;

  @media (max-width: ${scaling.xl}) {
    display: none;
  }
`;

const ChevronIcon = styled(FaChevronRight)<{ $expanded: boolean }>`
  transition: transform 0.2s ease;
  transform: rotate(${(p) => (p.$expanded ? '90deg' : '0deg')});
`;

const ChevronWrapper = styled.span`
  margin-left: auto;
  display: flex;
  align-items: center;
  font-size: 0.6rem;

  @media (max-width: ${scaling.xl}) {
    display: none;
  }
`;

const DirectLink = styled(NavLink)`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.5rem;
  border-radius: 6px;
  color: var(--thorium-nav-text);
  text-decoration: none;
  transition:
    background-color 0.15s ease,
    color 0.15s ease;

  @media (max-width: ${scaling.xl}) {
    justify-content: center;
  }

  &:hover {
    ${hoverStyle}
  }

  &.active {
    color: var(--thorium-highlight-text);
  }
`;

const DirectLinkLabel = styled.span`
  white-space: nowrap;
  font-size: 0.9rem;
  line-height: ${ICON_SIZE}px;

  @media (max-width: ${scaling.xl}) {
    display: none;
  }
`;

const SubItemList = styled.div<{ $expanded: boolean; $height: number }>`
  padding-left: 0.5rem;
  overflow: hidden;
  transition:
    max-height 0.25s ease,
    opacity 0.2s ease,
    visibility 0s ${(p) => (p.$expanded ? '0s' : '0.25s')};
  max-height: ${(p) => (p.$expanded ? `${p.$height}px` : '0')};
  opacity: ${(p) => (p.$expanded ? 1 : 0)};
  visibility: ${(p) => (p.$expanded ? 'visible' : 'hidden')};
`;

// Measured wrapper for the submenu items. Its scrollHeight drives the animated
// max-height, so the top gap that separates the submenu from the category row lives
// here (not on SubItemList) to keep the last item from being clipped.
const SubItemInner = styled.div`
  padding-top: 0.35rem;
`;

const SubItemLink = styled(NavLink)`
  display: flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.5rem;
  margin-bottom: 0.15rem;
  border-radius: 6px;
  color: var(--thorium-nav-text);
  text-decoration: none;
  font-size: 0.9rem;
  transition:
    background-color 0.15s ease,
    color 0.15s ease;

  // Keep the icon at full size in the fixed-width sidebar; a long label must not
  // squeeze it to zero. The label truncates instead (see SubItemLabel).
  & > svg {
    flex-shrink: 0;
  }

  &:hover {
    ${hoverStyle}
  }

  &.active {
    color: var(--thorium-highlight-text);
  }
`;

const SubItemLabel = styled.span`
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
  line-height: ${ICON_SIZE - 4}px;

  @media (max-width: ${scaling.xl}) {
    display: none;
  }
`;

// "Show more" / "Show less" toggle row. Mirrors SubItemLink sizing but is a real button
// (not a link) so keyboard users get Enter/Space activation and focus handling for free.
const SecondaryToggleRow = styled.button`
  display: flex;
  align-items: center;
  gap: 0.4rem;
  width: 100%;
  padding: 0.3rem 0.5rem;
  margin-bottom: 0.15rem;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  text-align: left;
  background: none;
  color: var(--thorium-secondary-text);
  font-size: 0.9rem;
  transition:
    background-color 0.15s ease,
    color 0.15s ease;

  & > svg {
    flex-shrink: 0;
  }

  &:hover {
    ${hoverStyle}
  }
`;

const SecondaryToggleIcon = styled(FaChevronDown)<{ $expanded: boolean }>`
  transition: transform 0.2s ease;
  transform: rotate(${(p) => (p.$expanded ? '180deg' : '0deg')});
`;

const SecondaryToggleLabel = styled.span`
  white-space: nowrap;
  line-height: ${ICON_SIZE - 4}px;

  @media (max-width: ${scaling.xl}) {
    display: none;
  }
`;

const fadeIn = keyframes`
  from { opacity: 0; transform: translateX(-4px); }
  to   { opacity: 1; transform: translateX(0); }
`;

const FlyoutPanel = styled.div<{ $top: number; $left: number }>`
  position: fixed;
  top: ${(p) => p.$top}px;
  left: ${(p) => p.$left}px;
  z-index: 5000;
  min-width: 160px;
  padding: 0.5rem 0;
  background-color: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 6px;
  box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
  animation: ${fadeIn} 0.15s ease forwards;
  max-height: calc(100vh - ${CanvasMargin.top}px);
  overflow-y: auto;
`;

const FlyoutHeader = styled.div`
  padding: 0.25rem 0.75rem 0.5rem;
  font-size: 0.75rem;
  color: var(--thorium-secondary-text);
  text-transform: uppercase;
  letter-spacing: 0.05em;
`;

const FlyoutItem = styled(NavLink)`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.25rem 0.75rem;
  color: var(--thorium-text);
  text-decoration: none;
  white-space: nowrap;
  transition:
    background-color 0.15s ease,
    color 0.15s ease;

  &:hover {
    ${hoverStyle}
  }

  &.active {
    color: var(--thorium-highlight-text);
  }
`;

// Flyout variant of the show-more toggle. Labels always show (the flyout exists to
// surface labels when the sidebar is icon-only), so no responsive hiding here. A real
// button so keyboard users get activation and focus handling for free.
const FlyoutSecondaryToggle = styled.button`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
  padding: 0.25rem 0.75rem;
  border: none;
  cursor: pointer;
  text-align: left;
  background: none;
  color: var(--thorium-secondary-text);
  white-space: nowrap;
  transition:
    background-color 0.15s ease,
    color 0.15s ease;

  &:hover {
    ${hoverStyle}
  }
`;

const SidebarTooltipStyle = createGlobalStyle`
  .sidebar-nav-tooltip .tooltip-inner {
    color: var(--thorium-secondary-text);
  }
`;

// --- Sub-components ---

interface SidebarSubItemProps {
  item: NavSubItem;
}

const SidebarSubItem: React.FC<SidebarSubItemProps> = ({ item }) => {
  const Icon = item.icon;
  return (
    <OverlayTipRight tip={item.label} className="sidebar-nav-tooltip">
      <SubItemLink to={item.path} end>
        <Icon size={ICON_SIZE - 4} />
        <SubItemLabel>{item.label}</SubItemLabel>
      </SubItemLink>
    </OverlayTipRight>
  );
};

interface SecondaryToggleProps {
  expanded: boolean;
  onToggle: () => void;
  testId?: string;
}

// Inline "Show more" / "Show less" toggle. Label hides in icon-only mode like SubItemLabel.
const SecondaryToggle: React.FC<SecondaryToggleProps> = ({ expanded, onToggle, testId }) => (
  <SecondaryToggleRow
    type="button"
    data-testid={testId}
    onClick={(e) => {
      e.stopPropagation();
      onToggle();
    }}
  >
    <SecondaryToggleIcon size={ICON_SIZE - 8} $expanded={expanded} />
    <SecondaryToggleLabel>{expanded ? 'Show less' : 'Show more'}</SecondaryToggleLabel>
  </SecondaryToggleRow>
);

interface AnimatedSubItemListProps {
  expanded: boolean;
  primary: NavSubItem[];
  secondary?: NavSubItem[];
  showSecondary: boolean;
  onToggleSecondary: () => void;
  toggleTestId?: string;
}

const AnimatedSubItemList: React.FC<AnimatedSubItemListProps> = ({
  expanded,
  primary,
  secondary,
  showSecondary,
  onToggleSecondary,
  toggleTestId,
}) => {
  const innerRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(0);
  const hasSecondary = !!secondary && secondary.length > 0;

  // Recompute the animated max-height whenever the rendered item count changes
  // (primary/secondary counts or the show-more toggle). The toggle row lives inside
  // the measured div so its height is included.
  useEffect(() => {
    if (innerRef.current) {
      setHeight(innerRef.current.scrollHeight);
    }
  }, [primary.length, secondary?.length, showSecondary, expanded]);

  return (
    <SubItemList $expanded={expanded} $height={height}>
      <SubItemInner ref={innerRef}>
        {primary.map((child) => (
          <SidebarSubItem key={child.path} item={child} />
        ))}
        {hasSecondary && showSecondary && secondary.map((child) => <SidebarSubItem key={child.path} item={child} />)}
        {hasSecondary && <SecondaryToggle expanded={showSecondary} onToggle={onToggleSecondary} testId={toggleTestId} />}
      </SubItemInner>
    </SubItemList>
  );
};

interface PortalFlyoutProps {
  category: NavCategory;
  anchorRef: React.RefObject<HTMLDivElement | null>;
  panelRef: React.RefObject<HTMLDivElement | null>;
  showSecondary: boolean;
  onToggleSecondary: () => void;
  // Called when a flyout link is selected, so the overlay closes + resets after navigating.
  onSelect: () => void;
  toggleTestId?: string;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

const PortalFlyout: React.FC<PortalFlyoutProps> = ({
  category,
  anchorRef,
  panelRef,
  showSecondary,
  onToggleSecondary,
  onSelect,
  toggleTestId,
  onMouseEnter,
  onMouseLeave,
}) => {
  const [pos, setPos] = useState(() => {
    if (anchorRef.current) {
      const rect = anchorRef.current.getBoundingClientRect();
      return { top: rect.top, left: rect.right + FLYOUT_GAP };
    }
    return { top: -9999, left: -9999 };
  });

  useLayoutEffect(() => {
    if (anchorRef.current) {
      const rect = anchorRef.current.getBoundingClientRect();
      setPos({ top: rect.top, left: rect.right + FLYOUT_GAP });
    }
  }, []);

  const handleToggleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onToggleSecondary();
  };

  const portalTarget = document.getElementById('root') || document.body;
  const secondary = category.secondaryChildren;
  const hasSecondary = !!secondary && secondary.length > 0;

  const renderFlyoutItem = (child: NavSubItem) => {
    const ChildIcon = child.icon;
    return (
      <FlyoutItem key={child.path} to={child.path} end onClick={onSelect}>
        <ChildIcon size={ICON_SIZE - 4} />
        {child.label}
      </FlyoutItem>
    );
  };

  return createPortal(
    <FlyoutPanel ref={panelRef} $top={pos.top} $left={pos.left} onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
      <FlyoutHeader>{category.label}</FlyoutHeader>
      {category.children!.map(renderFlyoutItem)}
      {hasSecondary && showSecondary && secondary.map(renderFlyoutItem)}
      {hasSecondary && (
        <FlyoutSecondaryToggle type="button" data-testid={toggleTestId} onClick={handleToggleClick}>
          <SecondaryToggleIcon size={ICON_SIZE - 8} $expanded={showSecondary} />
          {showSecondary ? 'Show less' : 'Show more'}
        </FlyoutSecondaryToggle>
      )}
    </FlyoutPanel>,
    portalTarget,
  );
};

interface SidebarCategoryProps {
  category: NavCategory;
  expanded: boolean;
  onToggle: (label: string) => void;
  showFlyout: boolean;
  onFlyoutEnter: (label: string) => void;
  onFlyoutLeave: () => void;
  onFlyoutClose: () => void;
}

const SidebarCategory: React.FC<SidebarCategoryProps> = ({
  category,
  expanded,
  onToggle,
  showFlyout,
  onFlyoutEnter,
  onFlyoutLeave,
  onFlyoutClose,
}) => {
  const rowRef = useRef<HTMLDivElement>(null);
  const flyoutPanelRef = useRef<HTMLDivElement>(null);
  const location = useLocation();

  const hasChildren = category.children && category.children.length > 0;
  const secondaryChildren = category.secondaryChildren;
  const hasSecondary = !!secondaryChildren && secondaryChildren.length > 0;
  const Icon = category.icon;

  // Session-only, per-category. Auto-reveal when the current route is a hidden entity.
  const isSecondaryActive = hasSecondary && secondaryChildren.some((child) => location.pathname.startsWith(child.path));
  const [showSecondary, setShowSecondary] = useState(isSecondaryActive);
  const toggleSecondary = useCallback(() => setShowSecondary((prev) => !prev), []);

  useEffect(() => {
    if (isSecondaryActive) setShowSecondary(true);
  }, [isSecondaryActive]);

  // Keep the hover flyout open across the layout shrink from "Show less". Collapsing
  // moves the panel out from under a stationary cursor, which React reports as a
  // mouseleave on both the panel AND this category (portal events propagate through
  // the React tree). While pinned we ignore those leaves and instead close only when
  // the pointer genuinely moves outside the flyout/anchor (pointermove effect below).
  const flyoutPinnedRef = useRef(false);
  // Last real cursor position; a shrink emits a synthetic pointermove at the same
  // coordinates, which must not be treated as the user moving away.
  const lastPointerRef = useRef<{ x: number; y: number } | null>(null);

  const isChildActive = hasChildren && (category.children!.some((child) => location.pathname.startsWith(child.path)) || isSecondaryActive);
  const isSelfActive = !hasChildren && category.path === location.pathname;

  const handleMouseEnter = useCallback(() => {
    if (hasChildren) onFlyoutEnter(category.label);
  }, [hasChildren, onFlyoutEnter, category.label]);

  const handleMouseLeave = useCallback(() => {
    if (!hasChildren) return;
    if (flyoutPinnedRef.current) return; // stay open while pinned across a collapse
    onFlyoutLeave();
  }, [hasChildren, onFlyoutLeave]);

  // Pin the flyout when its "Show more/less" toggle is used, so the ensuing shrink
  // doesn't close it. onFlyoutEnter cancels any pending hide before the re-layout.
  const handleFlyoutToggle = useCallback(() => {
    flyoutPinnedRef.current = true;
    onFlyoutEnter(category.label);
    toggleSecondary();
  }, [onFlyoutEnter, category.label, toggleSecondary]);

  // Definitively dismiss the flyout (click-outside, item select, pointer fully left): unpin, reset
  // "Show more" to its route-derived default so it reopens collapsed, and close immediately. Note
  // `showSecondary` is shared with the inline list, so this also collapses an inline Show-more choice.
  const closeFlyout = useCallback(() => {
    flyoutPinnedRef.current = false;
    setShowSecondary(isSecondaryActive);
    onFlyoutClose();
  }, [isSecondaryActive, onFlyoutClose]);

  // Reset the pin once the flyout is actually gone so it doesn't linger next time.
  useEffect(() => {
    if (!showFlyout) flyoutPinnedRef.current = false;
  }, [showFlyout]);

  // Close a pinned flyout once the pointer really leaves it and its anchor. Only registered while the
  // flyout is shown (a pin can only happen from inside a shown flyout), so at most one category has an
  // active document listener at a time instead of every category listening on every pointer move.
  useEffect(() => {
    if (!showFlyout) return;
    const isInside = (rect: DOMRect | undefined, x: number, y: number) =>
      !!rect && x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;

    const handlePointerMove = (e: PointerEvent) => {
      const last = lastPointerRef.current;
      if (last && last.x === e.clientX && last.y === e.clientY) return; // ignore synthetic moves
      lastPointerRef.current = { x: e.clientX, y: e.clientY };

      if (!flyoutPinnedRef.current) return;
      const panelRect = flyoutPanelRef.current?.getBoundingClientRect();
      const anchorRect = rowRef.current?.getBoundingClientRect();
      if (!isInside(panelRect, e.clientX, e.clientY) && !isInside(anchorRect, e.clientX, e.clientY)) {
        closeFlyout();
      }
    };

    document.addEventListener('pointermove', handlePointerMove);
    return () => document.removeEventListener('pointermove', handlePointerMove);
  }, [showFlyout, closeFlyout]);

  // Close (and reset) a shown flyout when the user clicks outside both the panel and its anchor row.
  // The flyout renders through a portal into #root, but the panel node is a real DOM child there, so
  // `contains` works. Anchor clicks are excluded — they switch to inline mode via the render guard.
  useEffect(() => {
    if (!showFlyout) return;
    const handleMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (flyoutPanelRef.current?.contains(target) || rowRef.current?.contains(target)) return;
      closeFlyout();
    };
    document.addEventListener('mousedown', handleMouseDown);
    return () => document.removeEventListener('mousedown', handleMouseDown);
  }, [showFlyout, closeFlyout]);

  if (!hasChildren) {
    return (
      <CategoryContainer>
        <OverlayTipRight tip={category.label} className="sidebar-nav-tooltip">
          <DirectLink to={category.path!} end={category.path === '/'}>
            <Icon size={ICON_SIZE} />
            <DirectLinkLabel>{category.label}</DirectLinkLabel>
          </DirectLink>
        </OverlayTipRight>
      </CategoryContainer>
    );
  }

  return (
    <CategoryContainer data-testid={`category-${category.label}`} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}>
      <CategoryRow ref={rowRef} $active={isChildActive || isSelfActive} onClick={() => onToggle(category.label)}>
        <Icon size={ICON_SIZE} />
        <CategoryLabel>{category.label}</CategoryLabel>
        <ChevronWrapper>
          <ChevronIcon $expanded={expanded} />
        </ChevronWrapper>
      </CategoryRow>

      <AnimatedSubItemList
        expanded={expanded}
        primary={category.children!}
        secondary={secondaryChildren}
        showSecondary={showSecondary}
        onToggleSecondary={toggleSecondary}
        toggleTestId={`secondary-toggle-${category.label}`}
      />

      {showFlyout && !expanded && (
        <PortalFlyout
          category={category}
          anchorRef={rowRef}
          panelRef={flyoutPanelRef}
          showSecondary={showSecondary}
          onToggleSecondary={handleFlyoutToggle}
          onSelect={closeFlyout}
          toggleTestId={`flyout-secondary-toggle-${category.label}`}
          onMouseEnter={handleMouseEnter}
          onMouseLeave={handleMouseLeave}
        />
      )}
    </CategoryContainer>
  );
};

interface SidebarProps {
  userInfo: UserInfo;
}

const Sidebar: React.FC<SidebarProps> = ({ userInfo }) => {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set());
  const [activeFlyout, setActiveFlyout] = useState<string | null>(null);
  const flyoutHideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const role = userInfo?.role as unknown as RoleKey;

  const toggleCategory = useCallback((label: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(label)) {
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
  }, []);

  const handleFlyoutEnter = useCallback((label: string) => {
    if (flyoutHideTimer.current) {
      clearTimeout(flyoutHideTimer.current);
      flyoutHideTimer.current = null;
    }
    setActiveFlyout(label);
  }, []);

  const handleFlyoutLeave = useCallback(() => {
    if (flyoutHideTimer.current) {
      clearTimeout(flyoutHideTimer.current);
    }
    flyoutHideTimer.current = setTimeout(() => setActiveFlyout(null), FLYOUT_DELAY_MS);
  }, []);

  // Immediate close (no hover-bridge delay) for deliberate dismissals — click-outside and item select.
  const handleFlyoutClose = useCallback(() => {
    if (flyoutHideTimer.current) {
      clearTimeout(flyoutHideTimer.current);
      flyoutHideTimer.current = null;
    }
    setActiveFlyout(null);
  }, []);

  const visibleItems = NAV_ITEMS.filter((item) => {
    if (item.adminOnly && role !== RoleKey.Admin) return false;
    return true;
  });

  return (
    <NavPanel>
      <SidebarTooltipStyle />
      {userInfo?.role &&
        visibleItems.map((category) => (
          <SidebarCategory
            key={category.label}
            category={category}
            expanded={expandedCategories.has(category.label)}
            onToggle={toggleCategory}
            showFlyout={activeFlyout === category.label}
            onFlyoutEnter={handleFlyoutEnter}
            onFlyoutLeave={handleFlyoutLeave}
            onFlyoutClose={handleFlyoutClose}
          />
        ))}
    </NavPanel>
  );
};

const SidebarColumn = () => {
  const { userInfo } = useAuth();
  if (userInfo && userInfo.token) {
    return (
      <SideCol>
        <RequireAuth>
          <Sidebar userInfo={userInfo} />
        </RequireAuth>
      </SideCol>
    );
  } else {
    return null;
  }
};

export default SidebarColumn;
