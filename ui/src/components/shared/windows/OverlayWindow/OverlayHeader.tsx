import styled from 'styled-components';

const OverlayHeader = styled.div<{ $zindex: number }>`
  display: flex;
  cursor: move;
  position: absolute;
  padding: 4px 4px 4px 10px;
  max-height: 36px; // maxHeight forces wrapping of extra things instead of increase in header height
  word-break: break-all; // breakup text at any point to ensure content doesn't overlap
  overflow-y: clip; // hide wrapped header content
  width: 100%; // width should match window dimensions
  font-size: 20px;
  border-bottom: solid 1px var(--thorium-panel-border);
  z-index: ${(props) => props.$zindex || 'auto'};
`;

export default OverlayHeader;
