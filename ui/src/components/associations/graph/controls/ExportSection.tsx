import React from 'react';

import { exportJPEG, exportPNG } from '../export';
import type { GraphSectionProps } from './types';
import { MenuList, MenuItem } from './Toolbar.styled';

const ExportSection: React.FC<GraphSectionProps> = ({ graphId, graphInstance }) => (
  <MenuList>
    <MenuItem onClick={() => exportPNG(graphId, graphInstance)}>PNG</MenuItem>
    <MenuItem onClick={() => exportJPEG(graphId, graphInstance)}>JPEG</MenuItem>
  </MenuList>
);

export default ExportSection;
