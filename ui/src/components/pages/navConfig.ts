import { IconType } from 'react-icons';
import {
  FaUpload,
  FaSearch,
  FaLayerGroup,
  FaFolderOpen,
  FaFolder,
  FaSitemap,
  FaCube,
  FaUsers,
  FaUser,
  FaCog,
  FaChartLine,
  FaTools,
  FaUserShield,
  FaCodeBranch,
  FaTachometerAlt,
  FaExclamationTriangle,
} from 'react-icons/fa';
import { FaHardDrive, FaFolderTree } from 'react-icons/fa6';
import { MdBusinessCenter } from 'react-icons/md';
// project imports
import { getBrowsingPathByEntity } from '@components/entities/browsing/EntityBrowsingRoutes';
import { Entities } from '@models/entities/entities';
import SigmaIcon from '@components/shared/icons/SigmaIcon';

export type NavIcon = IconType | React.ComponentType<{ size?: number }>;

export interface NavSubItem {
  label: string;
  icon: NavIcon;
  path: string;
}

export interface NavCategory {
  label: string;
  icon: NavIcon;
  path?: string;
  children?: NavSubItem[];
  adminOnly?: boolean;
}

export const NAV_ITEMS: NavCategory[] = [
  { label: 'Search', icon: FaSearch, path: '/' },
  { label: 'Analyze', icon: FaUpload, path: '/analyze' },
  {
    label: 'Browse',
    icon: FaLayerGroup,
    children: [
      { label: 'Files', icon: FaFolderOpen, path: getBrowsingPathByEntity(Entities.File) },
      { label: 'File Systems', icon: FaFolderTree, path: getBrowsingPathByEntity(Entities.FileSystem) },
      { label: 'Repos', icon: FaCodeBranch, path: getBrowsingPathByEntity(Entities.Repo) },
      { label: 'Collections', icon: FaFolder, path: getBrowsingPathByEntity(Entities.Collection) },
      { label: 'Devices', icon: FaHardDrive, path: getBrowsingPathByEntity(Entities.Device) },
      { label: 'Vendors', icon: MdBusinessCenter, path: getBrowsingPathByEntity(Entities.Vendor) },
      { label: 'Sigma Rules', icon: SigmaIcon, path: getBrowsingPathByEntity(Entities.SigmaRule) },
    ],
  },
  {
    label: 'Tools',
    icon: FaTools,
    children: [
      { label: 'Pipelines', icon: FaSitemap, path: '/pipelines' },
      { label: 'Images', icon: FaCube, path: '/images' },
      { label: 'Stats', icon: FaChartLine, path: '/stats' },
    ],
  },
  { label: 'Groups', icon: FaUsers, path: '/groups' },
  {
    label: 'Admin',
    icon: FaUserShield,
    adminOnly: true,
    children: [
      { label: 'Users', icon: FaUser, path: '/users' },
      { label: 'Settings', icon: FaCog, path: '/settings' },
    ],
  },
];
