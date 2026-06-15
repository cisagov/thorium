import styled from 'styled-components';

const OverlayBody = styled.div<{ $zindex: number; $hasHeader: boolean }>`
  margin-top: ${(props) => (props.$hasHeader ? '40px' : '0')};
  padding: 4px;
  z-index: ${(props) => props.$zindex || 'auto'};
  position: absolute;
  /* the body is a flex column and does NOT scroll itself — its content is expected to fill the
     height and own its own scroll (single scroll owner). this prevents scrollbars stacking with
     an inner viewer's scroll (double scrollbars). */
  display: flex;
  flex-direction: column;
  overflow: hidden;
  width: 100%;
  height: 100%;
  max-height: ${(props) => (props.$hasHeader ? 'calc(100% - 40px)' : '100%')};
`;

export default OverlayBody;
