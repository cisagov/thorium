import styled from 'styled-components';
import { Accordion } from 'react-bootstrap';

// project imports
import { BUTTON_BAR_MARGIN } from '@components/shared/buttons/tokens';

export const SECTION_COL_WIDTH = '500px';
export const SECTION_GAP = '6px';
export const DIVIDER_GAP = '4px';
export const COMBINED_WIDTH = `calc(${SECTION_COL_WIDTH} * 2 + ${SECTION_GAP} + ${DIVIDER_GAP})`;

export const CompactBody = styled(Accordion.Body)`
  padding-top: 0.25rem;
`;

export const ColumnsContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${SECTION_GAP};

  @media (min-width: 1200px) {
    flex-direction: row;
  }
`;

export const LeftColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${SECTION_GAP};

  @media (min-width: 1200px) {
    flex: 1;
    align-items: flex-end;
  }
`;

export const RightColumn = styled.div`
  display: flex;
  flex-direction: column;
  gap: ${SECTION_GAP};

  @media (min-width: 1200px) {
    flex: 1;
    align-items: flex-start;
  }
`;

export const ColumnCard = styled.div`
  padding: 8px 12px;
  background: var(--thorium-panel-bg);
  width: 100%;

  @media (min-width: 1200px) {
    max-width: ${SECTION_COL_WIDTH};
  }
`;

export const RightColumnCard = styled(ColumnCard)`
  position: relative;

  @media (min-width: 1200px) {
    padding-left: calc(${DIVIDER_GAP} + 12px);
    margin-left: ${DIVIDER_GAP};

    &::before {
      content: '';
      position: absolute;
      left: 0;
      top: 4px;
      bottom: 4px;
      width: 1px;
      background-color: var(--thorium-highlight-panel-border);
      border-radius: 1px;
    }
  }
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
