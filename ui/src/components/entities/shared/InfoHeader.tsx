import styled from 'styled-components';

const InfoHeader = styled.div<{ $bold?: boolean }>`
  display: flex;
  justify-content: flex-end;
  align-items: center;
  width: 16%;
  flex: 0 0 auto;
  min-width: 160px;
  margin: auto 0;
  gap: 0.5em;
  font-weight: ${(p) => (p.$bold ? 700 : 'normal')};
`;

export default InfoHeader;
