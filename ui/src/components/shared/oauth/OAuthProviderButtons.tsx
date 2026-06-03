// spec: ./SPEC.md
import styled from 'styled-components';

// project imports
import { getProviderMeta, sortProviders } from './providerMeta';
import Button from '@components/shared/buttons/Button';
import { ButtonSize, ButtonVariant } from '@components/shared/buttons/types';

interface OAuthProviderButtonsProps {
  /// Provider names from `GET /api/oauth/`
  providers: string[];
  /// Called with the chosen provider name when a button is clicked
  onSelect: (provider: string) => void;
  /// Disable all buttons (e.g. while a redirect is in flight)
  disabled?: boolean;
}

const ButtonStack = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  width: 100%;
`;

// Neutral, theme-following outline (the filled Secondary variant maps to a fixed light-blue that
// doesn't track the theme). The brand icon supplies the only color.
const ProviderButton = styled(Button)`
  width: 100%;
  border: 1px solid var(--thorium-panel-border);
  color: var(--thorium-text);
`;

const BrandIcon = styled.span<{ $color: string }>`
  color: ${({ $color }) => $color};
  display: inline-flex;
  align-items: center;
  font-size: 1.1em;
`;

/**
 * Renders one "Sign in with X" button per configured OAuth provider, sorted for stable order.
 * Renders nothing when there are no providers, so non-OAuth instances show only the password form.
 */
const OAuthProviderButtons: React.FC<OAuthProviderButtonsProps> = ({ providers, onSelect, disabled }) => {
  if (!providers || providers.length === 0) {
    return null;
  }
  return (
    <ButtonStack>
      {sortProviders(providers).map((provider) => {
        const meta = getProviderMeta(provider);
        const Icon = meta.Icon;
        return (
          <ProviderButton
            key={provider}
            variant={ButtonVariant.Ghost}
            size={ButtonSize.Large}
            disabled={disabled}
            onClick={() => onSelect(provider)}
          >
            <BrandIcon $color={meta.brandColor}>
              <Icon />
            </BrandIcon>
            Sign in with {meta.label}
          </ProviderButton>
        );
      })}
    </ButtonStack>
  );
};

export default OAuthProviderButtons;
