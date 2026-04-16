import * as THREE from 'three';

// project imports
import CollectionSVG from '@assets/icons/collection.svg?raw';
import DeviceSVG from '@assets/icons/device.svg?raw';
import FileSVG from '@assets/icons/file.svg?raw';
import FileGrowableSVG from '@assets/icons/file-add.svg?raw';
import FileSystemSVG from '@assets/icons/filesystem.svg?raw';
import FolderSVG from '@assets/icons/folder.svg?raw';
import FolderGrowableSVG from '@assets/icons/folder-add.svg?raw';
import RepoSVG from '@assets/icons/git.svg?raw';
import NetworkConnectionSVG from '@assets/icons/network-connection.svg?raw';
import OtherSVG from '@assets/icons/other.svg?raw';
import ProcessTreeSVG from '@assets/icons/process-tree.svg?raw';
import ProcessSVG from '@assets/icons/process.svg?raw';
import SigmaSVG from '@assets/icons/sigma.svg?raw';
import SigmaGrowableSVG from '@assets/icons/sigma-add.svg?raw';
import TagGrowableSVG from '@assets/icons/tag-add.svg?raw';
import TagSVG from '@assets/icons/tag.svg?raw';
import VendorGrowableSVG from '@assets/icons/vendor-add.svg?raw';
import VendorSVG from '@assets/icons/vendor.svg?raw';
import type { VisualState } from './types';
import { NodeType } from '@models/trees';

// default node state colors
const InitialNodeColor = '#00998C';
const GrowableNodeColor = '#64cc66';
// node type colors
const CollectionColor = '#8f30b8';
const DeviceColor = '#ed9624';
const FileColor = '#f1d592';
const FileSystemColor = '#8f30b8';
const FolderColor = '#D2B48C';
const NetworkConnectionColor = '#acc22e';
const OtherColor = '#cacfca';
const RepoColor = '#f03c2e';
const TagColor = '#427d8c';
const RuleColor = '#c60d00';
const VendorColor = '#8f30b8';
const WindowsProcessColor = '#fa8072';
const WindowsProcessTreeColor = '#808000';

const NODE_COLORS: Record<NodeType, Record<VisualState, string>> = {
  Collection: { basic: CollectionColor, growable: GrowableNodeColor, initial: InitialNodeColor },
  Device: { basic: DeviceColor, growable: GrowableNodeColor, initial: InitialNodeColor },
  File: { basic: FileColor, growable: GrowableNodeColor, initial: InitialNodeColor },
  FileSystem: { basic: FileSystemColor, growable: GrowableNodeColor, initial: InitialNodeColor },
  Folder: { basic: FolderColor, growable: GrowableNodeColor, initial: InitialNodeColor },
  NetworkConnection: { basic: NetworkConnectionColor, growable: GrowableNodeColor, initial: InitialNodeColor },
  Other: { basic: OtherColor, growable: GrowableNodeColor, initial: InitialNodeColor },
  Repo: { basic: RepoColor, growable: GrowableNodeColor, initial: InitialNodeColor },
  SigmaRule: { basic: RuleColor, growable: GrowableNodeColor, initial: InitialNodeColor },
  Tag: { basic: TagColor, growable: GrowableNodeColor, initial: InitialNodeColor },
  Vendor: { basic: VendorColor, growable: GrowableNodeColor, initial: InitialNodeColor },
  WindowsProcess: { basic: WindowsProcessColor, growable: GrowableNodeColor, initial: InitialNodeColor },
  WindowsProcessTree: { basic: WindowsProcessTreeColor, growable: GrowableNodeColor, initial: InitialNodeColor },
};

export const getNodeColor = (nodeType: NodeType, visualState: VisualState): string => {
  return NODE_COLORS[nodeType]?.[visualState] ?? NODE_COLORS.Other.basic;
};

const SVG_MAP: Record<NodeType, Record<VisualState, string>> = {
  Collection: {
    basic: CollectionSVG.replace('#REPLACEME', getNodeColor(NodeType.Collection, 'basic')),
    growable: CollectionSVG.replace('#REPLACEME', getNodeColor(NodeType.Collection, 'growable')),
    initial: CollectionSVG.replace('#REPLACEME', getNodeColor(NodeType.Collection, 'initial')),
  },
  Device: {
    basic: DeviceSVG.replace('#REPLACEME', getNodeColor(NodeType.Device, 'basic')),
    growable: DeviceSVG.replace('#REPLACEME', getNodeColor(NodeType.Device, 'growable')),
    initial: DeviceSVG.replace('#REPLACEME', getNodeColor(NodeType.Device, 'initial')),
  },
  File: {
    basic: FileSVG.replace('#REPLACEME', getNodeColor(NodeType.File, 'basic')),
    growable: FileGrowableSVG.replace('#REPLACEME', getNodeColor(NodeType.File, 'growable')),
    initial: FileSVG.replace('#REPLACEME', getNodeColor(NodeType.File, 'initial')),
  },
  FileSystem: {
    basic: FileSystemSVG.replace('#REPLACEME', getNodeColor(NodeType.FileSystem, 'basic')),
    growable: FileSystemSVG.replace('#REPLACEME', getNodeColor(NodeType.FileSystem, 'growable')),
    initial: FileSystemSVG.replace('#REPLACEME', getNodeColor(NodeType.FileSystem, 'initial')),
  },
  Folder: {
    basic: FolderSVG.replace('#REPLACEME', getNodeColor(NodeType.Folder, 'basic')),
    growable: FolderGrowableSVG.replace('#REPLACEME', getNodeColor(NodeType.Folder, 'growable')),
    initial: FolderSVG.replace('#REPLACEME', getNodeColor(NodeType.Folder, 'initial')),
  },
  NetworkConnection: {
    basic: NetworkConnectionSVG.replace('#REPLACEME', getNodeColor(NodeType.NetworkConnection, 'basic')),
    growable: NetworkConnectionSVG.replace('#REPLACEME', getNodeColor(NodeType.NetworkConnection, 'growable')),
    initial: NetworkConnectionSVG.replace('#REPLACEME', getNodeColor(NodeType.NetworkConnection, 'initial')),
  },
  Other: {
    basic: OtherSVG.replace('#REPLACEME', getNodeColor(NodeType.Other, 'basic')),
    growable: OtherSVG.replace('#REPLACEME', getNodeColor(NodeType.Other, 'growable')),
    initial: OtherSVG.replace('#REPLACEME', getNodeColor(NodeType.Other, 'initial')),
  },
  Repo: {
    basic: RepoSVG.replace('#REPLACEME', getNodeColor(NodeType.Repo, 'basic')),
    growable: RepoSVG.replace('#REPLACEME', getNodeColor(NodeType.Repo, 'growable')),
    initial: RepoSVG.replace('#REPLACEME', getNodeColor(NodeType.Repo, 'initial')),
  },
  SigmaRule: {
    basic: SigmaSVG.replace('#REPLACEME', getNodeColor(NodeType.SigmaRule, 'basic')),
    growable: SigmaGrowableSVG.replace('#REPLACEME', getNodeColor(NodeType.SigmaRule, 'growable')),
    initial: SigmaSVG.replace('#REPLACEME', getNodeColor(NodeType.SigmaRule, 'initial')),
  },
  Tag: {
    basic: TagSVG.replace('#REPLACEME', getNodeColor(NodeType.Tag, 'basic')),
    growable: TagGrowableSVG.replace('#REPLACEME', getNodeColor(NodeType.Tag, 'growable')),
    initial: TagSVG.replace('#REPLACEME', getNodeColor(NodeType.Tag, 'initial')),
  },
  Vendor: {
    basic: VendorSVG.replace('#REPLACEME', getNodeColor(NodeType.Vendor, 'basic')),
    growable: VendorGrowableSVG.replace('#REPLACEME', getNodeColor(NodeType.Vendor, 'growable')),
    initial: VendorSVG.replace('#REPLACEME', getNodeColor(NodeType.Vendor, 'initial')),
  },
  WindowsProcess: {
    basic: ProcessSVG.replace('#REPLACEME', getNodeColor(NodeType.WindowsProcess, 'basic')),
    growable: ProcessSVG.replace('#REPLACEME', getNodeColor(NodeType.WindowsProcess, 'growable')),
    initial: ProcessSVG.replace('#REPLACEME', getNodeColor(NodeType.WindowsProcess, 'initial')),
  },
  WindowsProcessTree: {
    basic: ProcessTreeSVG.replace('#REPLACEME', getNodeColor(NodeType.WindowsProcessTree, 'basic')),
    growable: ProcessTreeSVG.replace('#REPLACEME', getNodeColor(NodeType.WindowsProcessTree, 'growable')),
    initial: ProcessTreeSVG.replace('#REPLACEME', getNodeColor(NodeType.WindowsProcessTree, 'initial')),
  },
};

export const getNodeSvg = (nodeType: NodeType, visualState: VisualState): string => {
  return SVG_MAP[nodeType]?.[visualState] ?? SVG_MAP.Other.basic;
};

const textureCache = new Map<string, THREE.Texture>();

export const svgToTexture = (svgString: string, size = 64): THREE.Texture => {
  const cacheKey = `${svgString}_${size}`;
  const cached = textureCache.get(cacheKey);
  if (cached) return cached;

  // build canvas
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;

  const context = canvas.getContext('2d')!;
  const img = new Image();
  const texture = new THREE.Texture(canvas);
  img.onload = () => {
    context.drawImage(img, 0, 0, size, size);
    texture.needsUpdate = true;
  };
  const dataUri = `data:image/svg+xml;base64,${btoa(svgString)}`;
  img.src = dataUri;

  textureCache.set(cacheKey, texture);
  return texture;
};

let cachedEdgeColor: string | null = null;

const computeEdgeColor = (): string => {
  const rootTheme = document.getElementById('root')?.getAttribute('theme');
  const theme = rootTheme ?? '';
  if (theme === 'Dark' || theme === 'Ocean') {
    return getComputedStyle(document.documentElement).getPropertyValue('--thorium-secondary-text').trim() || 'darkgray';
  }
  return 'darkgray';
};

if (typeof MutationObserver !== 'undefined') {
  const rootEl = document.getElementById('root');
  if (rootEl) {
    new MutationObserver(() => {
      cachedEdgeColor = null;
    }).observe(rootEl, { attributes: true, attributeFilter: ['theme'] });
  }
}

export const getEdgeColor = (): string => {
  if (cachedEdgeColor === null) {
    cachedEdgeColor = computeEdgeColor();
  }
  return cachedEdgeColor;
};
