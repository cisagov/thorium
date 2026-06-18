import styled, { css } from 'styled-components';

/// A small danger-colored validation message shown directly under an invalid field.
export const FieldError = styled.div`
  color: var(--thorium-danger-bg);
  font-size: 12px;
  margin-top: 2px;
`;

/// A reusable danger outline + glow for inputs/selects in an error state. Splice into a
/// styled input via `${({ $error }) => $error && errorOutline}` and gate with a `$error` prop.
export const errorOutline = css`
  border-color: var(--thorium-danger-bg);
  box-shadow: 0 0 0 2px var(--thorium-danger-bg);
`;
