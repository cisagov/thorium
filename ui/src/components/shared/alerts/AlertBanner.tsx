import React, { useState } from 'react';
import styled from 'styled-components';

export enum Severity {
  Error = 'error',
  Warning = 'warning',
  Info = 'info',
  Success = 'success',
}

interface AlertBannerProps {
  severity?: Severity;
  dismissible?: boolean;
  className?: string;
  children: React.ReactNode;
}

const SEVERITY_TOKENS: Record<Severity, { bg: string; border: string; text: string }> = {
  error: {
    bg: 'color-mix(in srgb, var(--thorium-danger, #e74c3c) 12%, var(--thorium-panel-bg, #1e2028))',
    border: 'var(--thorium-danger, #e74c3c)',
    text: 'var(--thorium-danger, #e74c3c)',
  },
  warning: {
    bg: 'color-mix(in srgb, var(--thorium-warning, #f0ad4e) 12%, var(--thorium-panel-bg, #1e2028))',
    border: 'var(--thorium-warning, #f0ad4e)',
    text: 'var(--thorium-warning, #f0ad4e)',
  },
  info: {
    bg: 'color-mix(in srgb, var(--thorium-info, #5bc0de) 12%, var(--thorium-panel-bg, #1e2028))',
    border: 'var(--thorium-info, #5bc0de)',
    text: 'var(--thorium-text, #e0e0e0)',
  },
  success: {
    bg: 'color-mix(in srgb, var(--thorium-success, #2ecc71) 12%, var(--thorium-panel-bg, #1e2028))',
    border: 'var(--thorium-success, #2ecc71)',
    text: 'var(--thorium-success, #2ecc71)',
  },
};

const Banner = styled.div<{ $severity: Severity }>`
  padding: 0.75rem 1rem;
  border: 1px solid ${(p) => SEVERITY_TOKENS[p.$severity].border};
  border-radius: 6px;
  background: ${(p) => SEVERITY_TOKENS[p.$severity].bg};
  color: ${(p) => SEVERITY_TOKENS[p.$severity].text};
  font-size: 0.85rem;
  line-height: 1.5;
  text-align: center;
  position: relative;

  code {
    background: color-mix(in srgb, var(--thorium-panel-border, #3a3d45) 40%, transparent);
    padding: 0.1em 0.35em;
    border-radius: 3px;
    font-size: 0.82em;
  }
`;

const DismissButton = styled.button`
  position: absolute;
  top: 0.5rem;
  right: 0.6rem;
  background: none;
  border: none;
  color: inherit;
  opacity: 0.6;
  cursor: pointer;
  font-size: 1.1rem;
  line-height: 1;
  padding: 0 0.2rem;

  &:hover {
    opacity: 1;
  }
`;

const AlertBanner: React.FC<AlertBannerProps> = ({ severity = Severity.Error, dismissible = false, className, children }) => {
  const [visible, setVisible] = useState(true);

  if (!visible) return null;

  return (
    <Banner $severity={severity} className={className}>
      {dismissible ? <DismissButton onClick={() => setVisible(false)}>&times;</DismissButton> : null}
      {children}
    </Banner>
  );
};

export default AlertBanner;
