import React from 'react';
import SigmaSVG from '@assets/icons/sigma.svg?raw';

interface SigmaIconProps {
  size?: number;
  color?: string;
  className?: string;
}

const SigmaIcon: React.FC<SigmaIconProps> = ({ size = 24, color = 'currentColor', className }) => {
  const svg = SigmaSVG.replace('#REPLACEME', color)
    .replace(/height="[^"]*"/, `height="${size}"`)
    .replace(/width="[^"]*"/, `width="${size}"`);

  return <span className={className} style={{ display: 'inline-flex' }} dangerouslySetInnerHTML={{ __html: svg }} />;
};

export default SigmaIcon;
