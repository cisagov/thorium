/**
 * Get the API URL path as a string
 * @returns {string} API URL path
 */
export function getApiUrl() {
  if (window.location.hostname == 'localhost') {
    return process.env.REACT_APP_API_URL;
  } else {
    return `${window.location.protocol}//${window.location.hostname}/api`;
  }
}

// Update url hash location with section string
// append -tab to end of sections, results get an optional -subsection
export function updateURLSection(section: string, subsection: string) {
  const updatedHash = subsection ? `#${section}-${subsection}` : `#${section}`;
  window.location.hash = updatedHash;
}
