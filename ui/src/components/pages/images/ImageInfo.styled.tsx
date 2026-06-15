import styled from 'styled-components';
import { Accordion } from 'react-bootstrap';

// project imports
import { BUTTON_BAR_MARGIN } from '@components/shared/buttons/tokens';

export const SECTION_COL_WIDTH_PX = 500;
export const SECTION_COL_WIDTH = `${SECTION_COL_WIDTH_PX}px`;
export const SECTION_GAP = '6px';
export const COMBINED_WIDTH = `calc(${SECTION_COL_WIDTH} * 2 + ${SECTION_GAP})`;

export const CompactBody = styled(Accordion.Body)`
  padding-top: 0.25rem;
`;

export const ColumnCard = styled.div`
  padding: 8px 12px;
  background: var(--thorium-panel-bg);
  width: 100%;
`;

export const FieldsRow = styled.div`
  padding: 8px 12px;
  background: var(--thorium-panel-bg);
  margin-bottom: ${SECTION_GAP};

  @media (min-width: 1200px) {
    padding: 0;
    background: none;
  }
`;

export const CenteredContent = styled.div`
  @media (min-width: 1200px) {
    max-width: ${COMBINED_WIDTH};
    margin: 0 auto;
    padding: 8px 12px;
    background: var(--thorium-panel-bg);
  }
`;

export const FormWrapper = styled.div`
  max-width: ${COMBINED_WIDTH};
  margin: 0 auto;
`;

export const ErrorRow = styled.div`
  display: flex;
  justify-content: center;
  margin-bottom: 8px;

  @media (min-width: 1200px) {
    max-width: ${COMBINED_WIDTH};
    margin: 0 auto 8px;
  }
`;

export const CenterRow = styled.div`
  display: flex;
  justify-content: center;
  margin: ${BUTTON_BAR_MARGIN} 0;
`;

export const ImageBansContainer = styled.div`
  max-height: 200px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;

  @media (min-width: 1200px) {
    max-width: ${COMBINED_WIDTH};
    margin-left: auto;
    margin-right: auto;
  }
`;
