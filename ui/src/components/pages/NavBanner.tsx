import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FaQuestion } from 'react-icons/fa';
import styled from 'styled-components';

// project imports
import { OverlayTipRight } from '@components/shared/overlay/tips';
import { clearTagDataFromLocalStorage } from '@utilities/tags';
import { useAuth } from '@utilities/auth';
import { getApiUrl } from '@utilities/url';

const StyledNavbar = styled.nav`
  display: flex;
  align-items: center;
  justify-content: flex-end;
  background-color: var(--thorium-nav-panel-bg);
  border-bottom: 0.05px groove var(--thorium-panel-border);
  right: 0;
  position: fixed;
  width: 100%;
  z-index: 4000;
  padding: 0.1rem;
  margin: 0;
`;

const HomeLink = styled.button`
  margin-right: auto;
  border-radius: 10px;
  padding: 0.5rem;
  background: none;
  border: none;
  cursor: pointer;
  color: inherit;

  &:hover {
    background-color: var(--thorium-highlight-panel-bg);
  }
`;

const NavActions = styled.div`
  display: flex;
  align-items: center;
  justify-content: flex-end;
`;

const DocsLink = styled.a`
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--thorium-nav-text);
  background-color: var(--thorium-nav-panel-bg);
  padding: 0.5rem;
  margin: 0 0.5rem;
  border-radius: 10px;
  text-decoration: none;

  &:hover {
    color: var(--thorium-highlight-text);
  }
`;

const DropdownContainer = styled.div`
  position: relative;
`;

const DropdownToggle = styled.button`
  color: var(--thorium-nav-text);
  background-color: var(--thorium-nav-panel-bg);
  padding: 0.25rem 0.5rem;
  border-radius: 10px;
  border: none;
  cursor: pointer;
  font-size: inherit;

  &:hover {
    color: var(--thorium-highlight-text);
  }
`;

const DropdownMenu = styled.div<{ $open: boolean }>`
  display: ${(p) => (p.$open ? 'block' : 'none')};
  position: absolute;
  right: 0;
  top: 100%;
  min-width: 140px;
  padding: 0.25rem 0;
  margin-top: 0.25rem;
  background-color: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 6px;
  box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
  z-index: 5000;
`;

const DropdownItem = styled.button`
  display: block;
  width: 100%;
  padding: 0.35rem 0.75rem;
  background: none;
  border: none;
  color: var(--thorium-text);
  text-align: left;
  cursor: pointer;
  text-decoration: none;
  font-size: inherit;

  &:hover {
    color: var(--thorium-highlight-text);
    background-color: var(--thorium-highlight-panel-bg);
  }
`;

const DropdownLink = styled(Link)`
  display: block;
  width: 100%;
  padding: 0.35rem 0.75rem;
  color: var(--thorium-text);
  text-decoration: none;
  font-size: inherit;

  &:hover {
    color: var(--thorium-highlight-text);
    background-color: var(--thorium-highlight-panel-bg);
  }
`;

const NavBanner = () => {
  const { userInfo, logout } = useAuth();
  const navigate = useNavigate();
  const apiURL = getApiUrl();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const handleLogout = useCallback(() => {
    clearTagDataFromLocalStorage();
    void logout().then(() => {
      void navigate('/');
      window.location.reload();
    });
  }, [logout, navigate]);

  const handleHomeClick = useCallback(() => {
    if (window.location.pathname.startsWith('/?') || window.location.pathname === '/') {
      window.location.href = '/';
    } else {
      void navigate('/');
    }
  }, [navigate]);

  useEffect(() => {
    if (!dropdownOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [dropdownOpen]);

  return (
    <StyledNavbar>
      <HomeLink onClick={handleHomeClick}>
        <OverlayTipRight tip={'Home'}>
          <img src="/ferris-scientist.png" alt="FerrisScientist" width="40px" />
        </OverlayTipRight>
      </HomeLink>
      <NavActions>
        <DocsLink href={`${apiURL}/docs/user/index.html`}>
          <FaQuestion size={22} />
        </DocsLink>
        {userInfo && userInfo.username && (
          <DropdownContainer ref={dropdownRef}>
            <DropdownToggle onClick={() => setDropdownOpen((prev) => !prev)}>@{userInfo.username}</DropdownToggle>
            <DropdownMenu $open={dropdownOpen}>
              <DropdownLink to="/profile" onClick={() => setDropdownOpen(false)}>
                Profile
              </DropdownLink>
              <DropdownItem onClick={handleLogout}>Logout</DropdownItem>
            </DropdownMenu>
          </DropdownContainer>
        )}
      </NavActions>
    </StyledNavbar>
  );
};

export default NavBanner;
