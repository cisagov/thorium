import { scaling } from '@styles';
import styled from 'styled-components';

export const IMAGE_FIELDS_MAX_WIDTH = '700px';
export const IMAGE_CREATE_MAX_WIDTH = '900px';

export const SectionRow = styled.div`
  display: flex;
  gap: 8px;
  align-items: flex-start;

  @media (max-width: ${scaling.xl}) {
    flex-direction: column;
    gap: 4px;
    justify-content: center;
    max-width: ${IMAGE_FIELDS_MAX_WIDTH};
    margin: 0 auto;
  }
`;

export const IndentCol = styled.div`
  flex: 0 0 auto;
  min-width: 20px;
`;

export const ValCol = styled.div`
  flex: 1;
  /* allow the value column to shrink below its content's intrinsic width so long
     values wrap inside the card instead of overflowing the accordion body */
  min-width: 0;
`;

export const EditSpacer = styled.div`
  display: none;
`;

// Fixed width for every section's title column so titles have room (the longest
// is "Output Collection") and all section fields line up at the same x regardless
// of title length. Used by both the edit (EditMiddle) and create (TitleCol) forms.
export const SECTION_TITLE_WIDTH = '190px';

export const EditMiddle = styled.div`
  flex: 0 0 ${SECTION_TITLE_WIDTH};
`;

export const EditFieldCol = styled.div`
  flex: 1;
`;

export const TitleCol = styled.div`
  flex: 0 0 ${SECTION_TITLE_WIDTH};

  @media (max-width: ${scaling.xl}) {
    flex: 0 0 auto;
    width: 100%;
  }
`;

export const FieldCol = styled.div`
  flex: 1;
  @media (max-width: ${scaling.xl}) {
    width: 100%;
  }
`;

export const FieldGroup = styled.div`
  margin-bottom: 8px;
`;

export const Label = styled.label`
  display: block;
  font-size: 13px;
  font-weight: 600;
  color: var(--thorium-secondary-text);
  margin-bottom: 4px;
`;

export const SwitchRow = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
`;

export const ImageFieldsWrapper = styled.div`
  max-width: ${IMAGE_FIELDS_MAX_WIDTH};
`;

export const ImageCreateWrapper = styled.div`
  max-width: ${IMAGE_CREATE_MAX_WIDTH};
  margin: 0 auto;
  @media (max-width: ${scaling.xl}) {
    max-width: ${IMAGE_FIELDS_MAX_WIDTH};
  }
`;

export const AdvancedHidden = styled.div<{ $hidden: boolean }>`
  visibility: ${({ $hidden }) => ($hidden ? 'hidden' : 'visible')};
  display: ${({ $hidden }) => ($hidden ? 'none' : 'block')};
`;
