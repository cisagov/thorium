import { describe, it, expect } from 'vitest';

// project imports
import { isAuthed, isNewUser, isVerifyEmail, OAuthMaybeAuthed } from './oauth';

describe('OAuthMaybeAuthed type guards', () => {
  const authed: OAuthMaybeAuthed = { Authed: { token: 'tok', expires: '2099-01-01T00:00:00Z' } };
  const verifyEmail: OAuthMaybeAuthed = { VerifyEmail: 'user@example.com' };
  const newUser: OAuthMaybeAuthed = { NewUser: 'session-token' };

  it('isAuthed is true only for the Authed arm', () => {
    expect(isAuthed(authed)).toBe(true);
    expect(isAuthed(verifyEmail)).toBe(false);
    expect(isAuthed(newUser)).toBe(false);
  });

  it('isVerifyEmail is true only for the VerifyEmail arm', () => {
    expect(isVerifyEmail(verifyEmail)).toBe(true);
    expect(isVerifyEmail(authed)).toBe(false);
    expect(isVerifyEmail(newUser)).toBe(false);
  });

  it('isNewUser is true only for the NewUser arm', () => {
    expect(isNewUser(newUser)).toBe(true);
    expect(isNewUser(authed)).toBe(false);
    expect(isNewUser(verifyEmail)).toBe(false);
  });

  it('narrows to the carried payload', () => {
    if (isAuthed(authed)) {
      expect(authed.Authed.token).toBe('tok');
    }
    if (isVerifyEmail(verifyEmail)) {
      expect(verifyEmail.VerifyEmail).toBe('user@example.com');
    }
    if (isNewUser(newUser)) {
      expect(newUser.NewUser).toBe('session-token');
    }
  });
});
