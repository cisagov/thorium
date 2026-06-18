import styled from 'styled-components';

// project imports
import { BUTTON_BAR_MARGIN } from '@components/shared/buttons/tokens';
import { SectionRow } from './shared.styled';

export const InfoRow = styled.div`
  display: flex;
  gap: 8px;
  align-items: flex-start;
`;

export const HeaderCol = styled.div`
  min-width: 130px;
  flex: 0 0 auto;
`;

export const DetailCol = styled.div`
  flex: 8;
`;

export const TriggerRow = styled.div`
  display: flex;
  gap: 8px;
  align-items: flex-start;
`;

export const TriggerIndent = styled.div`
  max-width: 15px;
  flex: 0.5;
`;

export const TriggerField = styled.div`
  min-width: 135px;
  flex: 1;
`;

export const TriggerValue = styled.div`
  flex: 20;
`;

/// The form wrapper for the pipeline info/edit view (replaces the react-bootstrap <Form>)
export const EditForm = styled.form`
  width: 100%;
`;

/// Spacing wrapper for the update error alert
export const AlertSpacer = styled.div`
  margin-bottom: 0.25rem;
`;

/// Centered row holding the view-mode toggle
export const ToggleBar = styled.div`
  display: flex;
  justify-content: center;
  margin-bottom: ${BUTTON_BAR_MARGIN};
`;

/// Spacing wrapper for the format toggle
export const FormatBar = styled.div`
  margin-bottom: ${BUTTON_BAR_MARGIN};
`;

/// A form section with spacing above it (edit mode)
export const SpacedSectionRow = styled(SectionRow)`
  margin-top: 0.5rem;
`;

/// An info row with spacing above it (view mode)
export const SpacedInfoRow = styled(InfoRow)`
  margin-top: 0.25rem;
`;

/// Spacing wrapper for the order update error alert
export const OrderErrorSpacer = styled.div`
  margin-top: ${BUTTON_BAR_MARGIN};
`;

/// The creator badge (replaces the react-bootstrap Badge with the bg-blue utility)
export const CreatorBadge = styled.span`
  display: inline-block;
  padding: 0.35em 0.65em;
  font-size: 0.75em;
  font-weight: 700;
  line-height: 1;
  color: #fff;
  background-color: #0066cc;
  border-radius: 0.375rem;
`;
