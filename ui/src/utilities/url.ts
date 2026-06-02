/** Get the API URL path as a string. */
export function getApiUrl() {
  if (window.location.hostname == 'localhost' && import.meta.env.THORIUM_API_URL) {
    return `${String(import.meta.env.THORIUM_API_URL).replace(/\/+$/, '')}/api`;
  }
  return `${window.location.protocol}//${window.location.hostname}/api`;
}

/**
 * Update the URL hash to reflect the active section (and optional subsection).
 *
 * Sets `window.location.hash` to `#<section>` or `#<section>-<subsection>` when a subsection is
 * given, so deep links and the back button track the in-page navigation state.
 *
 * @param section - The primary section identifier.
 * @param subsection - The subsection identifier; pass an empty string for none.
 */
export function updateURLSection(section: string, subsection: string) {
  const updatedHash = subsection ? `#${section}-${subsection}` : `#${section}`;
  window.location.hash = updatedHash;
}
