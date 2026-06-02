import * as THREE from 'three';

// project imports
import CrabSVG from '@assets/icons/crab.svg?raw';
import CollectionSVG from '@assets/icons/collection.svg?raw';
import CollectionGrowableSVG from '@assets/icons/collection-add.svg?raw';
import DeviceSVG from '@assets/icons/device.svg?raw';
import DeviceGrowableSVG from '@assets/icons/device-add.svg?raw';
import FileSVG from '@assets/icons/file.svg?raw';
import FileGrowableSVG from '@assets/icons/file-add.svg?raw';
import FileSystemSVG from '@assets/icons/filesystem.svg?raw';
import FileSystemGrowableSVG from '@assets/icons/filesystem-add.svg?raw';
import FolderSVG from '@assets/icons/folder.svg?raw';
import FolderGrowableSVG from '@assets/icons/folder-add.svg?raw';
import RepoSVG from '@assets/icons/git.svg?raw';
import RepoGrowableSVG from '@assets/icons/git-add.svg?raw';
import NetworkConnectionSVG from '@assets/icons/network-connection.svg?raw';
import NetworkConnectionGrowableSVG from '@assets/icons/network-connection-add.svg?raw';
import OtherSVG from '@assets/icons/other.svg?raw';
import OtherGrowableSVG from '@assets/icons/other-add.svg?raw';
import ProcessTreeSVG from '@assets/icons/process-tree.svg?raw';
import ProcessTreeGrowableSVG from '@assets/icons/process-tree-add.svg?raw';
import ProcessSVG from '@assets/icons/process.svg?raw';
import ProcessGrowableSVG from '@assets/icons/process-add.svg?raw';
import SigmaSVG from '@assets/icons/sigma.svg?raw';
import SigmaGrowableSVG from '@assets/icons/sigma-add.svg?raw';
import TagGrowableSVG from '@assets/icons/tag-add.svg?raw';
import TagSVG from '@assets/icons/tag.svg?raw';
import VendorGrowableSVG from '@assets/icons/vendor-add.svg?raw';
import VendorSVG from '@assets/icons/vendor.svg?raw';
import { VisualState } from './types';
import { NodeType } from '@models/trees';

const LIGHT_DARKEN_FACTOR = 0.5;

const darkenHex = (hex: string, factor: number): string => {
  const r = Math.round(parseInt(hex.slice(1, 3), 16) * factor);
  const g = Math.round(parseInt(hex.slice(3, 5), 16) * factor);
  const b = Math.round(parseInt(hex.slice(5, 7), 16) * factor);
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
};

let cachedIsLight: boolean | null = null;

const isLightTheme = (): boolean => {
  if (cachedIsLight === null) {
    const theme = document.getElementById('root')?.getAttribute('theme') ?? '';
    cachedIsLight = theme === 'Light' || theme === 'Crab';
  }
  return cachedIsLight;
};

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
  const color = NODE_COLORS[nodeType]?.[visualState] ?? NODE_COLORS.Other.basic;
  return isLightTheme() ? darkenHex(color, LIGHT_DARKEN_FACTOR) : color;
};

// raw SVG templates keyed by node type and visual state (still contain #REPLACEME)
const RAW_SVG_MAP: Record<NodeType, Record<VisualState, string>> = {
  Collection: { basic: CollectionSVG, growable: CollectionGrowableSVG, initial: CollectionSVG },
  Device: { basic: DeviceSVG, growable: DeviceGrowableSVG, initial: DeviceSVG },
  File: { basic: FileSVG, growable: FileGrowableSVG, initial: FileSVG },
  FileSystem: { basic: FileSystemSVG, growable: FileSystemGrowableSVG, initial: FileSystemSVG },
  Folder: { basic: FolderSVG, growable: FolderGrowableSVG, initial: FolderSVG },
  NetworkConnection: { basic: NetworkConnectionSVG, growable: NetworkConnectionGrowableSVG, initial: NetworkConnectionSVG },
  Other: { basic: OtherSVG, growable: OtherGrowableSVG, initial: OtherSVG },
  Repo: { basic: RepoSVG, growable: RepoGrowableSVG, initial: RepoSVG },
  SigmaRule: { basic: SigmaSVG, growable: SigmaGrowableSVG, initial: SigmaSVG },
  Tag: { basic: TagSVG, growable: TagGrowableSVG, initial: TagSVG },
  Vendor: { basic: VendorSVG, growable: VendorGrowableSVG, initial: VendorSVG },
  WindowsProcess: { basic: ProcessSVG, growable: ProcessGrowableSVG, initial: ProcessSVG },
  WindowsProcessTree: { basic: ProcessTreeSVG, growable: ProcessTreeGrowableSVG, initial: ProcessTreeSVG },
};

export const getNodeSvg = (nodeType: NodeType, visualState: VisualState): string => {
  const raw = RAW_SVG_MAP[nodeType]?.[visualState] ?? RAW_SVG_MAP.Other.basic;
  return raw.replaceAll('#REPLACEME', getNodeColor(nodeType, visualState));
};

const textureCache = new Map<string, THREE.Texture>();

export const svgToTexture = (svgString: string, size = 64): THREE.Texture => {
  const cacheKey = `${svgString}_${size}`;
  const cached = textureCache.get(cacheKey);
  if (cached) return cached;

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

// Navy used for edges in light themes so lines read clearly against the light
// graph background. Matches the nav menu ($snl-dark-blue-700, the Light theme's
// --thorium-nav-panel-bg); hardcoded rather than read from the var because Crab's
// nav-panel-bg is amber and we want a consistent blue across both light themes.
const LIGHT_EDGE_COLOR = '#00243e';

const computeEdgeColor = (): string => {
  // darkgray reads well on dark backgrounds but washes out on light ones, so
  // light themes get the high-contrast navy instead.
  if (isLightTheme()) {
    return LIGHT_EDGE_COLOR;
  }
  return getComputedStyle(document.documentElement).getPropertyValue('--thorium-secondary-text').trim() || 'darkgray';
};

if (typeof MutationObserver !== 'undefined') {
  const rootEl = document.getElementById('root');
  if (rootEl) {
    new MutationObserver(() => {
      cachedEdgeColor = null;
      cachedIsLight = null;
      textureCache.clear();
    }).observe(rootEl, { attributes: true, attributeFilter: ['theme'] });
  }
}

export const getEdgeColor = (): string => {
  if (cachedEdgeColor === null) {
    cachedEdgeColor = computeEdgeColor();
  }
  return cachedEdgeColor;
};

export const isCrabTheme = (): boolean => {
  const theme = document.getElementById('root')?.getAttribute('theme') ?? '';
  return theme === 'Crab';
};

const CRAB_PARTICLE_SIZE = 4;

export const buildCrabParticle = (): THREE.Mesh => {
  const texture = svgToTexture(CrabSVG, 64);
  const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false, side: THREE.DoubleSide });
  const geometry = new THREE.PlaneGeometry(CRAB_PARTICLE_SIZE, CRAB_PARTICLE_SIZE * (20 / 24));
  return new THREE.Mesh(geometry, material);
};
