import styled from 'styled-components';
import { FaExclamationTriangle } from 'react-icons/fa';

// project imports
import { ButtonVariant, ButtonSize } from '@components/shared/buttons';
import { SIZE_TOKENS, VARIANT_TOKENS, BUTTON_BAR_GAP } from '@components/shared/buttons/tokens';
import AlertBanner from '@components/shared/alerts/AlertBanner';

// spec: ./browsing.spec.md

export const HeaderActions = styled.div`
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: ${BUTTON_BAR_GAP};
  min-width: 180px;
  margin-right: 8px;
`;

export const HeaderBtn = styled.span<{ $variant: ButtonVariant }>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  transition:
    background 0.15s ease,
    filter 0.15s ease;
  padding: ${SIZE_TOKENS[ButtonSize.XSmall].padding};
  font-size: ${SIZE_TOKENS[ButtonSize.XSmall].fontSize};
  background: ${({ $variant }) => VARIANT_TOKENS[$variant].bg};
  border: 1px solid ${({ $variant }) => VARIANT_TOKENS[$variant].border};
  color: ${({ $variant }) => VARIANT_TOKENS[$variant].text};

  &:hover {
    filter: brightness(85%);
  }

  &:active {
    filter: brightness(80%);
  }
`;

export const BanWarningIcon = styled(FaExclamationTriangle)`
  color: var(--thorium-danger, #e74c3c);
  font-size: 0.85em;
`;

export const BansContainer = styled.div`
  max-height: 200px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

export const BanItem = styled(AlertBanner)`
  padding: 0.3rem 0.75rem;
  text-align: left;
  font-size: 0.82rem;
`;
