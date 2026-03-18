import styled from 'styled-components';

const OverlayBody = styled.div<{ $zindex: number }>`
  margin-top: 40px;
  padding: 4px;
  z-index: ${(props) => props.$zindex || 'auto'};
  position: absolute;
  overflow-y: auto;
  overflow-x: hidden;
  width: 100%;
  height: 100%;
  max-height: calc(100% - 40px);
`;

export default OverlayBody;
