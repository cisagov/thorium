import { describe, it, expect } from 'vitest';

// project imports
import { resendButtonLabel, tickCooldown } from './VerifyEmailNotice';

describe('tickCooldown', () => {
  it('decrements by one second', () => {
    expect(tickCooldown(30)).toBe(29);
    expect(tickCooldown(1)).toBe(0);
  });

  it('clamps at zero', () => {
    expect(tickCooldown(0)).toBe(0);
    expect(tickCooldown(-5)).toBe(0);
  });
});

describe('resendButtonLabel', () => {
  it('shows the sending state regardless of cooldown', () => {
    expect(resendButtonLabel(0, true)).toBe('Sending…');
    expect(resendButtonLabel(30, true)).toBe('Sending…');
  });

  it('shows a countdown while cooling down', () => {
    expect(resendButtonLabel(30, false)).toBe('Resend in 30s');
    expect(resendButtonLabel(1, false)).toBe('Resend in 1s');
  });

  it('shows the resend label when ready', () => {
    expect(resendButtonLabel(0, false)).toBe('Resend');
  });
});
