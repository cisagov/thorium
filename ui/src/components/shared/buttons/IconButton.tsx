import { forwardRef } from 'react';
import styled from 'styled-components';

// project imports
import Button from './Button';
import { ButtonVariant, ButtonSize } from './types';
import type { ButtonProps } from './types';
import { ICON_SIZE } from './tokens';

const StyledIconButton = styled(Button)<{ $iconSize: string; $round: boolean }>`
  width: ${({ $iconSize }) => $iconSize};
  height: ${({ $iconSize }) => $iconSize};
  padding: 0;
  border-radius: ${({ $round }) => ($round ? '50%' : '6px')};
`;

export interface IconButtonProps extends ButtonProps {
  $round?: boolean;
}

const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ variant = ButtonVariant.Icon, size = ButtonSize.Medium, $round = false, children, ...rest }, ref) => (
    <StyledIconButton ref={ref} variant={variant} size={size} $iconSize={ICON_SIZE[size ?? ButtonSize.Medium]} $round={$round} {...rest}>
      {children}
    </StyledIconButton>
  ),
);

IconButton.displayName = 'IconButton';

export default IconButton;
