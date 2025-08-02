import '@styles/main.scss';

export const createReactSelectStyles = (color: string, backgroundColor: string) => {
  const newStyleTemplate = {
    input: (base: any) => ({
      ...base,
      color: 'var(--thorium-secondary-text)',
      //backgroundColor: 'yellow'
    }),
    singleValue: (base: any) => ({
      ...base,
      color: 'var(--thorium-text)',
    }),
    control: (base: any, state: any) => ({
      ...base,
      color: 'white',
      background: 'var(--thorium-secondary-panel-bg)',
      borderColor: state.isFocused ? 'var(--thorium-highlight-panel-border)' : 'var(--thorium-panel-border)',
      boxShadow: state.isFocused ? null : null,
      '&:hover': {
        borderColor: state.isFocused ? 'var(--thorium-highlight-panel-border)' : 'var(--thorium-panel-border)',
      },
    }),
    menu: (base: any) => ({
      ...base,
      backgroundColor: 'var(--thorium-secondary-panel-bg)',
    }),
    menuList: (base: any) => ({
      ...base,
      backgroundColor: 'var(--thorium-secondary-panel-bg)',
    }),
    option: (base: any) => ({
      ...base,
      background: 'var(--thorium-secondary-panel-bg)',
      '&:hover': {
        background: backgroundColor,
      },
    }),
    multiValue: (provided: any) => ({
      ...provided,
      color: color,
      backgroundColor: backgroundColor,
    }),
    multiValueLabel: (provided: any) => ({
      ...provided,
      color: color,
      backgroundColor: backgroundColor,
    }),
  };
  return newStyleTemplate;
};
