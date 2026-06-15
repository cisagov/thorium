// spec: ./Button.spec.md

export enum ButtonVariant {
  Primary = 'primary',
  Secondary = 'secondary',
  Ok = 'ok',
  Danger = 'danger',
  Warning = 'warning',
  Info = 'info',
  Ghost = 'ghost',
  Icon = 'icon',
}

export enum ButtonSize {
  XSmall = 'xs',
  Small = 'sm',
  Medium = 'md',
  Large = 'lg',
}

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}
