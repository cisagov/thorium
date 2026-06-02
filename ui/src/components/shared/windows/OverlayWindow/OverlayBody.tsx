import styled from 'styled-components';

const OverlayBody = styled.div<{ $zindex: number; $hasHeader: boolean }>`
  margin-top: ${(props) => (props.$hasHeader ? '40px' : '0')};
  padding: 4px;
  z-index: ${(props) => props.$zindex || 'auto'};
  position: absolute;
  overflow-y: auto;
  overflow-x: hidden;
  width: 100%;
  height: 100%;
  max-height: ${(props) => (props.$hasHeader ? 'calc(100% - 40px)' : '100%')};
`;

export default OverlayBody;
