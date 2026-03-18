import React, { Fragment } from 'react';
import { NavLink } from 'react-router-dom';
import { Col, Row } from 'react-bootstrap';
import { FaCube, FaFolderOpen, FaFolder, FaSitemap, FaUsers, FaUser, FaUpload, FaCog, FaChartLine } from 'react-icons/fa';
import { FaHardDrive } from 'react-icons/fa6';
import { MdBusinessCenter } from 'react-icons/md';
import styled from 'styled-components';

// project imports
import { OverlayTipRight } from '@components/shared/overlay/tips';
import { RequireAuth, useAuth } from '@utilities/auth';
import { RoleKey, UserInfo } from '@models/users';
import { CanvasMargin } from '@styles';

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
  top: ${CanvasMargin.top};
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
          <SidebarItem to="/collections" short={<FaFolder size={25} />} full={'Collections'} />
          <SidebarItem to="/devices" short={<FaHardDrive size={25} />} full={'Devices'} />
          <SidebarItem to="/vendors" short={<MdBusinessCenter size={25} />} full={'Vendors'} />
          <SidebarItem to="/pipelines" short={<FaSitemap size={25} />} full={'Pipelines'} />
          <SidebarItem to="/images" short={<FaCube size={25} />} full={'Images'} />
          <SidebarItem to="/groups" short={<FaUsers size={25} />} full={'Groups'} />
          <SidebarItem to="/stats" short={<FaChartLine size={25} />} full={'Stats'} />
          {role == RoleKey.Admin && <SidebarItem to="/users" short={<FaUser size={25} />} full={'Users'} />}
          {role == RoleKey.Admin && <SidebarItem to="/settings" short={<FaCog size={25} />} full={'Settings'} />}
        </Fragment>
      )}
    </NavPanel>
  );
};

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

export default SidebarColumn;
