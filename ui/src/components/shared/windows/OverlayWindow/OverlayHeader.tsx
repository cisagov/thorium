import styled from 'styled-components';

const OverlayHeader = styled.div<{ $zindex: number; $locked?: boolean }>`
  display: flex;
  cursor: ${(props) => (props.$locked ? 'default' : 'move')};
  position: absolute;
  padding: 4px 4px 4px 10px;
  max-height: 36px;
  word-break: break-all;
  overflow-y: clip;
  width: 100%;
  font-size: 20px;
  background-color: var(--thorium-panel-bg);
  border-bottom: solid 1px var(--thorium-panel-border);
  z-index: ${(props) => props.$zindex || 'auto'};
`;

export default OverlayHeader;
