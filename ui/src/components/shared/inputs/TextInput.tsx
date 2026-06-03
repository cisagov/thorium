import styled from 'styled-components';

// Themed single-line text input. Shared styled replacement for react-bootstrap Form.Control,
// used by auth/OAuth forms and reusable elsewhere. Pass `$invalid` to highlight a failed-validation
// field with a danger-colored border.
const TextInput = styled.input<{ $invalid?: boolean }>`
  background-color: var(--thorium-secondary-panel-bg);
  color: var(--thorium-text);
  border: 1px solid ${({ $invalid }) => ($invalid ? 'var(--thorium-danger-bg)' : 'var(--thorium-panel-border)')};
  border-radius: 6px;
  padding: 0.5rem 0.75rem;
  font-size: 1rem;
  width: 100%;

  &:focus-visible {
    outline: none;
    border-color: ${({ $invalid }) => ($invalid ? 'var(--thorium-danger-bg)' : 'var(--thorium-highlight-panel-border)')};
  }

  &::placeholder {
    color: var(--thorium-secondary-text);
  }
`;

export default TextInput;
