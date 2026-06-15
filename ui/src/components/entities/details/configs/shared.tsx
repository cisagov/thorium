import styled from 'styled-components';

// spec: ../EntityDetails.spec.md

// Themed muted note shown under an edit-mode field to explain an API update limitation
// (e.g. create-only fields, values that can't be cleared). Uses --thorium-secondary-text so
// contrast stays correct across all four themes rather than Bootstrap's un-themed text-muted.
export const CreateOnlyNote = styled.small`
  display: block;
  margin-top: 0.25rem;
  color: var(--thorium-secondary-text);
`;
