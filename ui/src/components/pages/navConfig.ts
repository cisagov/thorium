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
  FaFlag,
  FaNetworkWired,
  FaMicrochip,
  FaFileCode,
  FaLaptopCode,
  FaHammer,
} from 'react-icons/fa';
import { FaHardDrive, FaFolderTree, FaDiagramProject } from 'react-icons/fa6';
import { MdBusinessCenter } from 'react-icons/md';
// project imports
import { getBrowsingPathByEntity } from '@components/entities/browsing/EntityBrowsingRoutes';
import { Entities } from '@models/entities/entities';
import SigmaIcon from '@components/shared/icons/SigmaIcon';

// spec: ./SPEC.md
// spec: ./Page.spec.md

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
  // Less-common items hidden behind a "Show more" toggle, rendered below `children`.
  secondaryChildren?: NavSubItem[];
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
      { label: 'Incidents', icon: FaExclamationTriangle, path: getBrowsingPathByEntity(Entities.Incident) },
    ],
    // Less-common entity types, revealed by the "Show more" toggle. Add new types here.
    secondaryChildren: [
      { label: 'Flags', icon: FaFlag, path: getBrowsingPathByEntity(Entities.Flag) },
      { label: 'Network Connections', icon: FaNetworkWired, path: getBrowsingPathByEntity(Entities.NetworkConnection) },
      { label: 'Windows Processes', icon: FaMicrochip, path: getBrowsingPathByEntity(Entities.WindowsProcess) },
      { label: 'Windows Process Trees', icon: FaDiagramProject, path: getBrowsingPathByEntity(Entities.WindowsProcessTree) },
      { label: 'Compiled Functions', icon: FaFileCode, path: getBrowsingPathByEntity(Entities.CompiledFunction) },
      { label: 'Decompiled Functions', icon: FaLaptopCode, path: getBrowsingPathByEntity(Entities.DecompiledFunction) },
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
  {
    label: 'Dashboards',
    icon: FaTachometerAlt,
    children: [
      // Build is the primary entry point, so it leads the Dashboards group ahead of specific dashboards
      { label: 'Build', icon: FaHammer, path: '/dashboard/build' },
      { label: 'Incident', icon: FaExclamationTriangle, path: '/dashboard/incident' },
    ],
  },
];
