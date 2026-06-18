import type { CSSObjectWithLabel } from 'react-select';
import '@styles/main.scss';

/**
 * Build a react-select `styles` config wired to Thorium's theme CSS variables.
 *
 * Most colors are driven by `--thorium-*` variables so the control follows the active theme; the
 * `color`/`backgroundColor` arguments only customize the multi-value "chip" and option hover.
 *
 * @param color - Text color applied to selected multi-value chips.
 * @param backgroundColor - Background color applied to multi-value chips and option hover.
 * @param error - When true, the control gets a danger border + glow to flag an invalid field.
 * @returns A styles object suitable for the `styles` prop of a react-select component.
 */
export const createReactSelectStyles = (color: string, backgroundColor: string, error: boolean = false) => {
  return {
    input: (base: CSSObjectWithLabel) => ({
      ...base,
      color: 'var(--thorium-secondary-text)',
    }),
    singleValue: (base: CSSObjectWithLabel) => ({
      ...base,
      color: 'var(--thorium-text)',
    }),
    control: (base: CSSObjectWithLabel, state: { isFocused: boolean }) => ({
      ...base,
      color: 'white',
      background: 'var(--thorium-secondary-panel-bg)',
      borderColor: error
        ? 'var(--thorium-danger-bg)'
        : state.isFocused
          ? 'var(--thorium-highlight-panel-border)'
          : 'var(--thorium-panel-border)',
      boxShadow: error ? '0 0 0 2px var(--thorium-danger-bg)' : 'none',
      '&:hover': {
        borderColor: error
          ? 'var(--thorium-danger-bg)'
          : state.isFocused
            ? 'var(--thorium-highlight-panel-border)'
            : 'var(--thorium-panel-border)',
      },
    }),
    menu: (base: CSSObjectWithLabel) => ({
      ...base,
      backgroundColor: 'var(--thorium-secondary-panel-bg)',
    }),
    menuList: (base: CSSObjectWithLabel) => ({
      ...base,
      backgroundColor: 'var(--thorium-secondary-panel-bg)',
    }),
    option: (base: CSSObjectWithLabel) => ({
      ...base,
      background: 'var(--thorium-secondary-panel-bg)',
      '&:hover': {
        background: backgroundColor,
      },
    }),
    multiValue: (provided: CSSObjectWithLabel) => ({
      ...provided,
      color: color,
      backgroundColor: backgroundColor,
    }),
    multiValueLabel: (provided: CSSObjectWithLabel) => ({
      ...provided,
      color: color,
      backgroundColor: backgroundColor,
    }),
  };
};
