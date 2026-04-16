import styled from 'styled-components';

export const TreeContainer = styled.div`
  .tree button[role='treeitem'] {
    display: flex;
    background: transparent;
    border: none;
    width: 100%;
    padding: 0 0 2px 0;
    border-radius: 8px;
    transition: background-color 0.2s ease;
    cursor: pointer;
    color: var(--thorium-text);
  }

  .tree button[role='treeitem']:hover {
    background-color: rgb(0, 102, 255, 0.1);
  }

  .treeitem {
    text-align: left;
    color: inherit;
    padding: 6px 10px;
    position: relative;
    display: inline-flex;
    align-items: center;
  }

  .tree button[role='treeitem']:focus {
    outline: none;
  }

  button:focus-visible .treeitem.focused,
  .treeitem.searchmatch.focused {
    outline: 2px solid var(--thorium-text);
  }

  .treeitem.drop {
    border-color: var(--selected-color);
    background-color: #e1f1f8;
  }

  .treeitem.searchmatch {
    background-color: #e1f8ff;
  }

  .treeitem.folder:before {
    content: '';
    display: inline-block;
    width: 0;
    height: 0;
    border-style: solid;
    border-width: 4px 0 4px 6px;
    border-color: transparent transparent transparent currentColor;
    opacity: 0.7;
    margin-right: 6px;
    flex-shrink: 0;
    transition: transform 0.1s ease-in-out;
  }

  .treeitem.folder.expanded:before {
    transform: rotate(90deg);
  }

  .treeitem:not(.folder) {
    padding-left: 24px;
  }

  .treeitem.selected:after {
    content: ' ';
    position: absolute;
    top: 5px;
    left: -2px;
    height: 24px;
    width: 4px;
    background-color: #0366d6;
    border-radius: 99px;
  }

  .treeitem.duplicate-highlight {
    outline: 2px dashed #e8a838;
    outline-offset: -2px;
    background-color: rgba(232, 168, 56, 0.12);
  }

  .treeitem.duplicate-highlight:hover {
    background-color: rgba(232, 168, 56, 0.2);
  }

  .outeritem {
    display: flex;
    align-items: center;
    gap: 2px;
  }
  .outeritem button:not([role='treeitem']) {
    padding: 2px 4px;
    height: 80%;
  }

  .node-type-icon {
    display: inline-block;
    width: 16px;
    height: 16px;
    margin-right: 12px;
    flex-shrink: 0;
  }

  .node-type-icon img {
    width: 100%;
    height: 100%;
  }

  .overlay-trigger {
    display: inline-flex;
    align-items: center;
  }

  .duplicate-indicator {
    display: inline-block;
    font-size: 0.6rem;
    padding: 0 4px;
    margin-left: 4px;
    border-radius: 3px;
    background-color: rgba(232, 168, 56, 0.25);
    color: #b07d1a;
    vertical-align: middle;
  }
`;
