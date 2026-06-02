import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { NavLink, useLocation } from 'react-router-dom';
import { FaChevronRight } from 'react-icons/fa';
import styled, { createGlobalStyle, css, keyframes } from 'styled-components';

// project imports
import { NAV_ITEMS, NavCategory, NavSubItem } from './navConfig';
import { OverlayTipRight } from '@components/shared/overlay/tips';
import { RequireAuth, useAuth } from '@utilities/auth';
import { RoleKey, UserInfo } from '@models/users';
import { CanvasMargin, scaling } from '@styles';

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

  &:hover {
    ${hoverStyle}
  }

  &.active {
    color: var(--thorium-highlight-text);
  }
`;

const SubItemLabel = styled.span`
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

interface AnimatedSubItemListProps {
  expanded: boolean;
  children: NavSubItem[];
}

const AnimatedSubItemList: React.FC<AnimatedSubItemListProps> = ({ expanded, children }) => {
  const innerRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(0);

  useEffect(() => {
    if (innerRef.current) {
      setHeight(innerRef.current.scrollHeight);
    }
  }, [children.length]);

  return (
    <SubItemList $expanded={expanded} $height={height}>
      <div ref={innerRef}>
        {children.map((child) => (
          <SidebarSubItem key={child.path} item={child} />
        ))}
      </div>
    </SubItemList>
  );
};

interface PortalFlyoutProps {
  category: NavCategory;
  anchorRef: React.RefObject<HTMLDivElement | null>;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

const PortalFlyout: React.FC<PortalFlyoutProps> = ({ category, anchorRef, onMouseEnter, onMouseLeave }) => {
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

  const portalTarget = document.getElementById('root') || document.body;

  return createPortal(
    <FlyoutPanel $top={pos.top} $left={pos.left} onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
      <FlyoutHeader>{category.label}</FlyoutHeader>
      {category.children!.map((child) => {
        const ChildIcon = child.icon;
        return (
          <FlyoutItem key={child.path} to={child.path} end>
            <ChildIcon size={ICON_SIZE - 4} />
            {child.label}
          </FlyoutItem>
        );
      })}
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
}

const SidebarCategory: React.FC<SidebarCategoryProps> = ({ category, expanded, onToggle, showFlyout, onFlyoutEnter, onFlyoutLeave }) => {
  const rowRef = useRef<HTMLDivElement>(null);
  const location = useLocation();

  const hasChildren = category.children && category.children.length > 0;
  const Icon = category.icon;

  const isChildActive = hasChildren && category.children!.some((child) => location.pathname.startsWith(child.path));
  const isSelfActive = !hasChildren && category.path === location.pathname;

  const handleMouseEnter = useCallback(() => {
    if (hasChildren) onFlyoutEnter(category.label);
  }, [hasChildren, onFlyoutEnter, category.label]);

  const handleMouseLeave = useCallback(() => {
    if (hasChildren) onFlyoutLeave();
  }, [hasChildren, onFlyoutLeave]);

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

      <AnimatedSubItemList expanded={expanded} children={category.children!} />

      {showFlyout && !expanded && (
        <PortalFlyout category={category} anchorRef={rowRef} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave} />
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
