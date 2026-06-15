import styled from 'styled-components';

// spec: ./SPEC.md

/// The header row above the seeded dashboard: incident title on the left, "Change incident" on the right.
export const IncidentHeader = styled.div`
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
  margin-bottom: 0.75rem;
`;

/// The selected incident's title.
export const IncidentTitle = styled.h2`
  color: var(--thorium-text);
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
`;

/// The subtle text button that clears the `?incident` param to return to the picker.
export const ChangeIncidentButton = styled.button`
  background: none;
  border: none;
  padding: 0;
  color: var(--thorium-link-text);
  font-size: 0.85rem;
  cursor: pointer;
  text-decoration: underline;

  &:hover {
    color: var(--thorium-highlight-text);
  }
`;

/// The card wrapping the incident picker when no incident is selected.
export const PickerCard = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
  max-width: 640px;
  margin: 2rem auto;
  padding: 1.5rem;
  background: var(--thorium-panel-bg);
  border: 1px solid var(--thorium-panel-border);
  border-radius: 8px;
`;

/// The picker heading.
export const PickerHeading = styled.h2`
  color: var(--thorium-text);
  margin: 0;
  font-size: 1.4rem;
  font-weight: 600;
`;

/// The short explanatory text under the picker heading.
export const PickerIntro = styled.p`
  color: var(--thorium-secondary-text);
  margin: 0;
  font-size: 0.9rem;
`;

/// The row holding the incident select control.
export const PickerSelectRow = styled.div`
  display: flex;
  flex-direction: column;
`;
