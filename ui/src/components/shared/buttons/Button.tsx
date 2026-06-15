import { forwardRef } from 'react';
import styled, { css } from 'styled-components';

// project imports
import { ButtonVariant, ButtonSize } from './types';
import type { ButtonProps } from './types';
import { SIZE_TOKENS, VARIANT_TOKENS } from './tokens';

// spec: ./Button.spec.md

const filledHoverCss = css`
  filter: brightness(85%);
`;

const iconHoverCss = css`
  color: var(--thorium-highlight-text);
`;

const StyledButton = styled.button<{ $variant: ButtonVariant; $size: ButtonSize }>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  text-decoration: none;
  transition:
    background 0.15s ease,
    border-color 0.15s ease,
    color 0.15s ease,
    box-shadow 0.15s ease,
    filter 0.15s ease,
    transform 0.1s ease;

  padding: ${({ $size }) => SIZE_TOKENS[$size].padding};
  font-size: ${({ $size }) => SIZE_TOKENS[$size].fontSize};

  background: ${({ $variant }) => VARIANT_TOKENS[$variant].bg};
  border: 1px solid ${({ $variant }) => VARIANT_TOKENS[$variant].border};
  color: ${({ $variant }) => VARIANT_TOKENS[$variant].text};

  &:hover:not(:disabled) {
    background: ${({ $variant }) => VARIANT_TOKENS[$variant].hoverBg};
    border-color: ${({ $variant }) => VARIANT_TOKENS[$variant].hoverBorder};
    ${({ $variant }) => ($variant === ButtonVariant.Icon ? iconHoverCss : $variant === ButtonVariant.Ghost ? '' : filledHoverCss)}
  }

  &:active:not(:disabled) {
    transform: scale(0.98);
    ${({ $variant }) => $variant !== ButtonVariant.Icon && $variant !== ButtonVariant.Ghost && 'filter: brightness(80%);'}
  }

  &:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px ${({ $variant }) => VARIANT_TOKENS[$variant].focusRing};
  }

  &:disabled {
    opacity: 0.45;
    cursor: not-allowed;
    filter: none;
  }
`;

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = ButtonVariant.Primary, size = ButtonSize.Medium, children, ...rest }, ref) => (
    <StyledButton ref={ref} $variant={variant} $size={size} type="button" {...rest}>
      {children}
    </StyledButton>
  ),
);

Button.displayName = 'Button';

export default Button;
