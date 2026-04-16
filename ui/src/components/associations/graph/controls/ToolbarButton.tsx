import React, { useRef, useState } from 'react';
import { Overlay, Popover, Tooltip } from 'react-bootstrap';

import type { SectionKey } from './types';
import { ToolbarIconButton, StyledPopover } from './Toolbar.styled';

interface ToolbarButtonProps {
  sectionKey: SectionKey;
  activeSection: SectionKey | null;
  onToggle: (key: SectionKey) => void;
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}

const ToolbarButton: React.FC<ToolbarButtonProps> = ({ sectionKey, activeSection, onToggle, icon, title, children }) => {
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [hovered, setHovered] = useState(false);
  const isOpen = activeSection === sectionKey;

  return (
    <>
      <ToolbarIconButton
        ref={buttonRef}
        $active={isOpen}
        onClick={() => onToggle(sectionKey)}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        {icon}
      </ToolbarIconButton>

      <Overlay target={buttonRef.current} show={!isOpen && hovered} placement="top">
        {(props) => (
          <Tooltip {...props} id={`tooltip-${sectionKey}`}>
            {title}
          </Tooltip>
        )}
      </Overlay>

      <Overlay target={buttonRef.current} show={isOpen} placement="top">
        {(props) => (
          <StyledPopover {...props} id={`popover-${sectionKey}`}>
            <Popover.Body>{children}</Popover.Body>
          </StyledPopover>
        )}
      </Overlay>
    </>
  );
};

export default ToolbarButton;
