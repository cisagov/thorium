import React from 'react';
import { IconType } from 'react-icons';
import {
  FaCodeBranch,
  FaCube,
  FaFileLines,
  FaFlag,
  FaFolder,
  FaFolderTree,
  FaHardDrive,
  FaLayerGroup,
  FaMicrochip,
  FaShieldHalved,
} from 'react-icons/fa6';
import { FaCode, FaExclamationTriangle, FaFileImport, FaPuzzlePiece, FaRegFileCode } from 'react-icons/fa';
import { MdBusinessCenter } from 'react-icons/md';
import { PiNetwork } from 'react-icons/pi';

// project imports
import { Entities, entityLabel } from '@models/entities';

/** Maps each entity type to a representative react-icon. */
const ENTITY_TYPE_ICONS: Record<Entities, IconType> = {
  [Entities.File]: FaFileLines,
  [Entities.Repo]: FaCodeBranch,
  [Entities.Device]: FaMicrochip,
  [Entities.Vendor]: MdBusinessCenter,
  [Entities.Collection]: FaLayerGroup,
  [Entities.FileSystem]: FaHardDrive,
  [Entities.Folder]: FaFolder,
  [Entities.WindowsProcessTree]: FaFolderTree,
  [Entities.WindowsProcess]: FaCube,
  [Entities.NetworkConnection]: PiNetwork,
  [Entities.SigmaRule]: FaShieldHalved,
  [Entities.Flag]: FaFlag,
  [Entities.Incident]: FaExclamationTriangle,
  [Entities.CompiledFunction]: FaCode,
  [Entities.DecompiledFunction]: FaRegFileCode,
  [Entities.PeSection]: FaPuzzlePiece,
  [Entities.PeImport]: FaFileImport,
  [Entities.Other]: FaCube,
};

interface EntityTypeIconProps {
  kind: Entities;
  size?: number;
  className?: string;
  title?: string;
}

/** Renders the icon for an entity type, falling back to a generic cube for unknown types. */
const EntityTypeIcon: React.FC<EntityTypeIconProps> = ({ kind, size = 16, className, title }) => {
  const Icon = ENTITY_TYPE_ICONS[kind] ?? FaCube;
  return <Icon size={size} className={className} title={title ?? entityLabel(kind)} aria-label={entityLabel(kind)} />;
};

export default EntityTypeIcon;
