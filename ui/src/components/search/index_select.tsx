import styled from 'styled-components';
import { DropdownButton, Dropdown } from 'react-bootstrap';

import { OverlayTipTop } from '@components';

// @ts-ignore
const IndexDropdown = styled(DropdownButton)`
  position: absolute;
  right: 0;
  top: 0;
  height: 100%;
  border: none;
  background: none;
  // ensure the button is above input text
  z-index: 1;

  // left border of the dropdown button
  .dropdown-toggle::before {
    content: '';
    position: absolute;
    left: 0;
    top: 12%; /* Adjust this value to control the gap at the top */
    bottom: 10%; /* Adjust this value to control the gap at the bottom */
    width: 3px;
    background-color: var(--thorium-nav-panel-bg); /* Border color */
    border-radius: 0; /* Ensure no rounded corners */
  }

  .dropdown-toggle {
    color: var(--thorium-text);
    height: 100%;
    border: none;
    background: none;
    padding: 0 10px;
    transition: none;
  }

  .dropdown-toggle:active {
    color: var(--thorium-nav-text);
    background-color: var(--thorium-nav-panel-bg);
  }

  .dropdown-menu.show + .dropdown-toggle,
  .dropdown-toggle[aria-expanded='true'] {
    color: var(--thorium-nav-text);
    background-color: var(--thorium-nav-panel-bg);
  }

  .dropdown-menu {
    right: 0;
    left: auto;
    color: var(--thorium-nav-text);
    background-color: var(--thorium-nav-panel-bg);
  }

  .dropdown-item:hover {
    color: var(--thorium-selected-text);
    background-color: var(--thorium-highlight-panel-bg);
  }
`;

interface IndexSelectProps {
  index: string; // selected index
  onChange: (selectedIndex: string | null) => void;
}

export const IndexSelect: React.FC<IndexSelectProps> = ({ index, onChange }) => {
  return (
    <OverlayTipTop tip={`Select which index to search (or search all indexes)`}>
      <IndexDropdown id="dropdown-basic-button" title={index} onSelect={(index) => onChange(index)}>
        <Dropdown.Item eventKey="All">All</Dropdown.Item>
        <Dropdown.Item eventKey="Results">Results</Dropdown.Item>
        <Dropdown.Item eventKey="Tags">Tags</Dropdown.Item>
      </IndexDropdown>
    </OverlayTipTop>
  );
};
