import { forwardRef } from 'react';
import styled from 'styled-components';

// project imports
// spec: ./Panel.spec.md
import { PanelVariant } from './types';
import type { PanelProps } from './types';
import { VARIANT_TOKENS } from './tokens';

const PanelRoot = styled.div<{ $variant: PanelVariant }>`
  position: relative;
  display: flex;
  flex-direction: column;
  min-width: 0;
  word-wrap: break-word;
  background-clip: border-box;

  background-color: ${({ $variant }) => VARIANT_TOKENS[$variant].bg};
  color: ${({ $variant }) => VARIANT_TOKENS[$variant].text};
  border: ${({ $variant }) => VARIANT_TOKENS[$variant].border};
  border-radius: ${({ $variant }) => VARIANT_TOKENS[$variant].borderRadius};
  overflow: ${({ $variant }) => VARIANT_TOKENS[$variant].overflow};
`;

const PanelHeader = styled.div`
  padding: 0.5rem 1rem;
  border-bottom: 1px solid var(--thorium-panel-border);
  background-color: inherit;
  color: inherit;

  &:first-child {
    border-top-left-radius: inherit;
    border-top-right-radius: inherit;
  }
`;

const PanelBody = styled.div`
  flex: 1 1 auto;
  padding: 1rem;
  color: inherit;
`;

const PanelTitle = styled.div`
  font-size: 1.25rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: inherit;
`;

type PanelComponent = React.ForwardRefExoticComponent<PanelProps & React.RefAttributes<HTMLDivElement>> & {
  Header: typeof PanelHeader;
  Body: typeof PanelBody;
  Title: typeof PanelTitle;
};

const Panel = forwardRef<HTMLDivElement, PanelProps>(({ variant = PanelVariant.Standard, children, ...rest }, ref) => (
  <PanelRoot ref={ref} $variant={variant} {...rest}>
    {children}
  </PanelRoot>
)) as PanelComponent;

Panel.displayName = 'Panel';
Panel.Header = PanelHeader;
Panel.Body = PanelBody;
Panel.Title = PanelTitle;

export default Panel;
