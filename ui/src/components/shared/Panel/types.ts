export enum PanelVariant {
  Standard = 'standard',
  Flush = 'flush',
  Result = 'result',
  Outlined = 'outlined',
}

export interface PanelProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: PanelVariant;
}
