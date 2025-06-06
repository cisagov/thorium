import { StylesheetJson } from 'cytoscape';

// Project imports
import FileSVG from '@assets/icons/file.svg';
import FileSearchSVG from '@assets/icons/file-add.svg';
import TagSVG from '@assets/icons/tag.svg';
import TagSearchSVG from '@assets/icons/tag-add.svg';
import RepoSVG from '@assets/icons/git.svg';
import VendorSVG from '@assets/icons/vendor.svg';
import VendorSearchSVG from '@assets/icons/vendor-add.svg';
import DeviceSVG from '@assets/icons/device.svg';
import OtherSVG from '@assets/icons/other.svg';

// replace hardcoded key in SVG with actual colors
export const DeviceGrowableIcon = DeviceSVG.replace('REPLACEME', '64cc66');
export const FileGrowableIcon = FileSearchSVG.replace('REPLACEME', '64cc66');
export const OtherGrowableIcon = OtherSVG.replace('REPLACEME', '64cc66');
export const RepoGrowableIcon = RepoSVG.replace('REPLACEME', '64cc66');
export const TagGrowableIcon = TagSearchSVG.replace('REPLACEME', '64cc66');
export const VendorGrowableIcon = VendorSearchSVG.replace('REPLACEME', '64cc66');
export const DeviceIcon = DeviceSVG.replace('REPLACEME', 'ed9624');
export const FileIcon = FileSVG.replace('REPLACEME', 'f1d592');
export const OtherIcon = OtherSVG.replace('REPLACEME', 'cacfca');
export const RepoIcon = RepoSVG.replace('REPLACEME', 'f03c2e');
export const TagIcon = TagSVG.replace('REPLACEME', '427d8c');
export const VendorIcon = VendorSVG.replace('REPLACEME', '8f30b8');
export const InitialDeviceIcon = DeviceSVG.replace('REPLACEME', '00998C');
export const InitialFileIcon = FileSVG.replace('REPLACEME', '00998C');
export const InitialOtherIcon = OtherSVG.replace('REPLACEME', '00998C');
export const InitialRepoIcon = RepoSVG.replace('REPLACEME', '00998C');
export const InitialTagIcon = TagSVG.replace('REPLACEME', '00998C');
export const InitialVendorIcon = VendorSVG.replace('REPLACEME', '00998C');

// get edge color based on theme
const getEdgeColor = (style: CSSStyleDeclaration) => {
  const rootTheme = document.getElementById('root')?.getAttribute('theme');
  const theme = rootTheme ? rootTheme : '';
  // dark/colored modes
  if (theme == 'Dark' || theme == 'Ocean') {
    return style.getPropertyValue('--thorium-text-secondary');
  }
  // light mode
  return 'darkgray';
};

export const buildStyleSheet = (rootNodes: string[]): StylesheetJson => {
  const computedStyle = getComputedStyle(document.documentElement);

  return [
    {
      selector: 'node',
      style: {
        'min-zoomed-font-size': 2,
        'background-fit': 'cover',
        'background-image-opacity': 1,
        'background-opacity': 0,
        'text-valign': 'bottom',
        'text-halign': 'center',
        width: 'data(diameter)',
        height: 'data(diameter)',
        color: computedStyle.getPropertyValue('--thorium-text'),
      },
    },
    {
      selector: 'node[type="Tag"]',
      style: {},
    },
    {
      selector: 'node[type="Repo"]',
      style: {},
    },
    {
      selector: 'edge',
      style: {
        width: 1,
        'line-color': getEdgeColor(computedStyle),
        'target-arrow-color': getEdgeColor(computedStyle),
        color: computedStyle.getPropertyValue('--thorium-text'),
        'min-zoomed-font-size': 3,
        'target-arrow-shape': 'triangle',
        'arrow-scale': 0.7,
        'curve-style': 'bezier',
      },
    },
    {
      selector: '.bidirectional',
      style: {
        width: 1,
        'source-arrow-color': getEdgeColor(computedStyle),
        'source-arrow-shape': 'triangle',
      },
    },
    {
      selector: 'edge:selected',
      style: {
        'line-color': '#007BFF',
        'target-arrow-color': '#007BFF',
      },
    },
    {
      selector: '.basic-repo',
      style: {
        'background-image': `url(${RepoIcon})`,
      },
    },
    {
      selector: '.growable-repo',
      style: {
        'background-image': `url(${RepoGrowableIcon})`,
      },
    },
    {
      selector: '.basic-file',
      style: {
        'background-image': `url(${FileIcon})`,
      },
    },
    {
      selector: '.growable-file',
      style: {
        'background-image': `url(${FileGrowableIcon})`,
      },
    },
    {
      selector: '.basic-other',
      style: {
        'background-image': `url(${OtherIcon})`,
      },
    },
    {
      selector: '.growable-other',
      style: {
        'background-image': `url(${OtherGrowableIcon})`,
      },
    },
    {
      selector: '.basic-tag',
      style: {
        'background-image': `url(${TagIcon})`,
      },
    },
    {
      selector: '.growable-tag',
      style: {
        'background-image': `url(${TagGrowableIcon})`,
      },
    },
    {
      selector: '.basic-vendor',
      style: {
        'background-image': `url(${VendorIcon})`,
      },
    },
    {
      selector: '.growable-vendor',
      style: {
        'background-image': `url(${VendorGrowableIcon})`,
      },
    },
    {
      selector: '.basic-device',
      style: {
        'background-image': `url(${DeviceIcon})`,
      },
    },
    {
      selector: '.growable-device',
      style: {
        'background-image': `url(${DeviceGrowableIcon})`,
      },
    },
    {
      selector: '.has-node-label',
      style: {
        label: 'data(label)',
      },
    },
    {
      selector: '.has-edge-label',
      style: {
        label: 'data(label)',
        color: getEdgeColor(computedStyle),
        opacity: 0.7,
      },
    },
    {
      selector: `.initial-file`,
      style: {
        'background-image': `url(${InitialFileIcon})`,
      },
    },
    {
      selector: `.initial-other`,
      style: {
        'background-image': `url(${InitialOtherIcon})`,
      },
    },
    {
      selector: `.initial-repo`,
      style: {
        'background-image': `url(${InitialRepoIcon})`,
      },
    },
    {
      selector: `.initial-tag`,
      style: {
        'background-image': `url(${InitialTagIcon})`,
      },
    },
    {
      selector: `.initial-vendor`,
      style: {
        'background-image': `url(${InitialVendorIcon})`,
      },
    },
    {
      selector: `.initial-device`,
      style: {
        'background-image': `url(${InitialDeviceIcon})`,
      },
    },
  ];
};
