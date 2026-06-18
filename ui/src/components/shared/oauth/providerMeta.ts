import { IconType } from 'react-icons';
import { FaApple, FaGithub, FaGitlab, FaGoogle, FaMicrosoft } from 'react-icons/fa6';
import { SiAuth0, SiKeycloak, SiOkta, SiOpenid } from 'react-icons/si';

/// Display metadata for an OAuth provider login button.
export type ProviderMeta = {
  /// The raw provider name (config key) from the API
  name: string;
  /// Human-friendly label rendered on the button
  label: string;
  /// Brand icon component
  Icon: IconType;
  /// Brand accent color for the icon
  brandColor: string;
};

type KnownMeta = Omit<ProviderMeta, 'name'>;

// Known providers, keyed by lowercased provider name. Colors are chosen to stay legible on both
// light and dark Thorium themes (avoid pure black/white).
const KNOWN_PROVIDERS: Record<string, KnownMeta> = {
  google: { label: 'Google', Icon: FaGoogle, brandColor: '#4285F4' },
  github: { label: 'GitHub', Icon: FaGithub, brandColor: '#8B949E' },
  gitlab: { label: 'GitLab', Icon: FaGitlab, brandColor: '#FC6D26' },
  microsoft: { label: 'Microsoft', Icon: FaMicrosoft, brandColor: '#00A4EF' },
  azure: { label: 'Microsoft', Icon: FaMicrosoft, brandColor: '#00A4EF' },
  apple: { label: 'Apple', Icon: FaApple, brandColor: '#999999' },
  okta: { label: 'Okta', Icon: SiOkta, brandColor: '#3F59E4' },
  auth0: { label: 'Auth0', Icon: SiAuth0, brandColor: '#EB5424' },
  keycloak: { label: 'Keycloak', Icon: SiKeycloak, brandColor: '#4D9FE0' },
};

// Fallback for any provider name not in the known table — a generic OpenID badge.
const GENERIC_ICON: IconType = SiOpenid;
const GENERIC_COLOR = '#F78C40';

/// Convert an arbitrary provider config key (e.g. `corp-okta`, `my_idp`) into a display label.
function titleCase(name: string): string {
  return name
    .replace(/[_-]+/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Resolve display metadata for a provider name (case-insensitive), falling back to a generic
 * OpenID icon and a title-cased label for unknown providers.
 *
 * @param name - The provider name returned by `GET /api/oauth/`.
 * @returns Metadata to render the provider's login button.
 */
export function getProviderMeta(name: string): ProviderMeta {
  const known = KNOWN_PROVIDERS[name.trim().toLowerCase()];
  if (known) {
    return { name, ...known };
  }
  return { name, label: titleCase(name) || name, Icon: GENERIC_ICON, brandColor: GENERIC_COLOR };
}

/**
 * Return a stable, alphabetically-sorted copy of provider names.
 *
 * The API returns providers from an unordered map, so button order would otherwise be
 * nondeterministic between requests.
 *
 * @param names - The provider names to sort.
 * @returns A new, sorted array (the input is not mutated).
 */
export function sortProviders(names: string[]): string[] {
  return [...names].sort((a, b) => a.localeCompare(b));
}
