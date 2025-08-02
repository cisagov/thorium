import React, { Fragment } from 'react';
import { Link, NavLink, useNavigate } from 'react-router-dom';
import { Col, Nav, Navbar, NavDropdown, Row } from 'react-bootstrap';
import { FaCube, FaFolderOpen, FaQuestion, FaSitemap, FaUsers, FaUser, FaUpload, FaCog, FaChartLine } from 'react-icons/fa';
import styled from 'styled-components';

// project imports
import { OverlayTipRight } from '@components';
import { RequireAuth, useAuth, getApiUrl } from '@utilities';
import { RoleKey, UserInfo } from '@models';

const NavBanner = () => {
  const { userInfo, logout } = useAuth();
  const navigate = useNavigate();
  const apiURL = getApiUrl();

  // call auth logout and redirect to login
  const handleLogout = () => {
    // re-render navbar to remove username
    logout().then(() => {
      navigate('/');
      // force reload of site on logout
      window.location.reload();
    });
  };

  // go to the root
  const handleHomeClick = () => {
    if (window.location.pathname.startsWith('/?') || window.location.pathname == '/') {
      // if we're already at the root, force a page reload to reload the search page
      window.location.href = '/';
    } else {
      // otherwise just navigate there
      navigate('/');
    }
  };

  return (
    <Navbar className="navbar-banner panel d-flex justify-content-end">
      <Nav.Link className="home-item" onClick={handleHomeClick}>
        <OverlayTipRight tip={'Home'}>
          <img src="/ferris-scientist.png" alt="FerrisScientist" width="40px" />
        </OverlayTipRight>
      </Nav.Link>
      <Nav className="d-flex justify-content-end">
        {
          <Navbar.Brand className="navbanner-item mx-2 px-2 pb-3" href={`${apiURL}/docs/user/index.html`}>
            <FaQuestion className="mt-3" size={22} />
          </Navbar.Brand>
        }
        {userInfo && userInfo.username && (
          <NavDropdown align="end" className="navbanner-item" title={`@${userInfo.username}`}>
            <NavDropdown.Item as={Link} to="/profile">
              Profile
            </NavDropdown.Item>
            <NavDropdown.Item onClick={() => handleLogout()}>Logout</NavDropdown.Item>
          </NavDropdown>
        )}
      </Nav>
    </Navbar>
  );
};

interface SidebarItemProps {
  to: string; // path to navigate to
  short: React.JSX.Element; // page icon element
  full: string; // full page name
}

const SidebarItem: React.FC<SidebarItemProps> = ({ to, short, full }) => {
  return (
    <NavLink
      to={to} // no-decoration
      className={(navData) => (navData.isActive ? 'activeNavLink' : 'navLink')}
    >
      <Row className="reduce-sidebar">
        <Col xs="auto" className="short">
          <OverlayTipRight tip={full}>{short}</OverlayTipRight>
        </Col>
      </Row>
      <Row className="expand-sidebar">
        <Col xs="auto" className="short">
          {short}
        </Col>
        <Col>{full}</Col>
      </Row>
    </NavLink>
  );
};

const NavPanel = styled.div`
  z-index: 0;
  left: 0;
  top: 47px;
  padding: 0.5rem 1rem;
  position: fixed;
  height: 100%;
  border-right: 0.05px groove var(--thorium-panel-border);
  color: var(--thorium-nav-text);
  background-color: var(--thorium-nav-panel-bg);
`;

interface SidebarProps {
  userInfo: UserInfo;
}

const Sidebar: React.FC<SidebarProps> = ({ userInfo }) => {
  const role = userInfo?.role as unknown as RoleKey;
  return (
    <NavPanel className="pt-4">
      {userInfo?.role && (
        <Fragment>
          <SidebarItem to="/upload" short={<FaUpload size={25} />} full={'Upload'} />
          <SidebarItem to="/files" short={<FaFolderOpen size={25} />} full={'Files'} />
          <SidebarItem to="/pipelines" short={<FaSitemap size={25} />} full={'Pipelines'} />
          <SidebarItem to="/images" short={<FaCube size={25} />} full={'Images'} />
          <SidebarItem to="/groups" short={<FaUsers size={25} />} full={'Groups'} />
          {role == RoleKey.Admin && <SidebarItem to="/users" short={<FaUser size={25} />} full={'Users'} />}
          {role == RoleKey.Admin && <SidebarItem to="/stats" short={<FaChartLine size={25} />} full={'Stats'} />}
          {role == RoleKey.Admin && <SidebarItem to="/settings" short={<FaCog size={25} />} full={'Settings'} />}
        </Fragment>
      )}
    </NavPanel>
  );
};

// @ts-ignore
const SideCol = styled(Col)`
  flex: 1 !important;
  flex-basis: 170px !important;
  flex-shrink: 0 !important;
  flex-grow: 0 !important;
`;

const SidebarColumn = () => {
  const { userInfo } = useAuth();
  if (userInfo && userInfo.token) {
    return (
      <SideCol className="sidebar-column">
        <RequireAuth>
          <Sidebar userInfo={userInfo} />
        </RequireAuth>
      </SideCol>
    );
  } else {
    return null;
  }
};

export { SidebarColumn, NavBanner };
