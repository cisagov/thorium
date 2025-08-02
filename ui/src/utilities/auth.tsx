import React, { createContext, JSX, useContext, useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { authUserPass, createUser, logout, whoami } from '@thorpi';
import { UserInfo, RoleKey } from '@models';

type AuthContextType = {
  userInfo: UserInfo | null;
  token: string | undefined;
  refreshUserInfo: (force?: boolean) => Promise<void>;
  checkCookie: () => Promise<unknown>;
  login: (username: string, password: string, handleError: (error: string) => void) => Promise<unknown>;
  logout: () => Promise<unknown>;
  register: (username: string, password: string, handleError: (error: string) => void, email?: string, role?: string) => Promise<unknown>;
  revoke: () => Promise<unknown>;
  impersonate: (userToken: string, tokenExpires: string) => Promise<void>;
};

// auth context to store info about auth state across app
const authContext = createContext<AuthContextType | undefined>(undefined);

// get document cookie by name
function getCookie(name: string) {
  const cookieArr = document.cookie.split(';');

  for (let i = 0; i < cookieArr.length; i++) {
    const cookiePair = cookieArr[i].trim().split('=');

    if (cookiePair[0] === name) {
      return decodeURIComponent(cookiePair[1]);
    }
  }
  return undefined;
}

function buildCookie(token: string, expiration: string) {
  return `THORIUM_TOKEN=${token}; Secure; SameSite=Strict expires=${expiration}; path=/; domain: ${location.hostname}`;
}

/*
 * Thorium auth hooks for login, logout and token revocation
 */
function useAuthProvider() {
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [token, setToken] = useState(getCookie('THORIUM_TOKEN'));
  // set time of last userInfo update
  const [lastUpdateDate, setLastUpdateDate] = useState(Date.now());
  // options for set/get of a secure cookie
  const getUserInfo = async () => {
    // get user details
    if (token != undefined) {
      whoami().then((response) => {
        if (response) {
          setUserInfo(response);
          setToken(response.token);
          setLastUpdateDate(Date.now());
        } else {
          document.cookie = 'THORIUM_TOKEN=; max-age=0;';
          setToken('');
        }
      });
    }
  };

  useEffect(() => {
    // update theme if userInfo changes
    if (userInfo?.settings?.theme) {
      const root = document.getElementById('root');
      // automatic theme will use browser defaults
      if (userInfo.settings['theme'] == 'Automatic') {
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
          root?.setAttribute('theme', 'Dark');
        } else {
          root?.setAttribute('theme', 'Light');
        }
      } else {
        root?.setAttribute('theme', userInfo.settings.theme);
      }
    }
  }, [userInfo]);

  useEffect(() => {
    getUserInfo();
    // need to run whoami to get user info after login
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  return {
    userInfo,
    token,
    // verify userInfo is not out-of-date
    async refreshUserInfo(force = false) {
      // check if userInfo is fresher than 60 seconds (60k msec)
      if (force || Date.now() - lastUpdateDate > 60000) {
        // refresh user info
        getUserInfo();
      }
      return;
    },
    // validate cookie with request to Thorium whoami route
    // a 401 response will clear cookie
    checkCookie() {
      return new Promise((resolve) => {
        if (getCookie('THORIUM_TOKEN')) {
          whoami().then((response) => {
            if (response) {
              setUserInfo(response);
              setToken(response.token);
              setLastUpdateDate(Date.now());
              resolve(response as UserInfo);
            } else {
              document.cookie = 'THORIUM_TOKEN=; max-age=0;';
              setUserInfo(null);
              setToken(undefined);
              resolve(null);
            }
          });
        }
      });
    },
    // login via password to get Thorium token
    login(username: string, password: string, handleError: (error: string) => void) {
      return new Promise((resolve) => {
        authUserPass(username, password, handleError).then((authResp) => {
          if (authResp) {
            // set cookie with name THORIUM_TOKEN
            document.cookie = buildCookie(authResp.token, authResp.expires);
            // set user's Thorium token
            setToken(authResp.token);
            resolve(true);
          } else {
            resolve(false);
          }
        });
      });
    },
    // remove token and clear user info on logout
    logout() {
      return new Promise((resolve) => {
        setToken(undefined);
        setUserInfo(null);
        document.cookie = 'THORIUM_TOKEN=; max-age=0;';
        resolve(true);
      });
    },
    // register with Thorium
    register(username: string, password: string, handleError: (error: string) => void, email = 'thorium@sandia.gov', role = 'User') {
      return new Promise((resolve) => {
        createUser(username, email, password, role, handleError).then((authResp) => {
          if (authResp) {
            // set cookie with name THORIUM_TOKEN
            document.cookie = buildCookie(authResp.token, authResp.expires);
            // set user's Thorium token
            setToken(authResp.token);
            resolve(true);
          } else {
            resolve(false);
          }
        });
      });
    },
    // revoke token and clear cookie user info from session
    revoke() {
      return new Promise((resolve) => {
        /**
         * Submits a token revocation request to the Thorium API
         * @returns {object} promise for async revocation action
         */
        async function handleRevoke() {
          const response = await logout();
          if (response?.status == 200) {
            resolve(true);
          } else {
            resolve(false);
          }
        }
        handleRevoke().then(() => {
          setToken(undefined);
          setUserInfo(null);
          document.cookie = 'THORIUM_TOKEN=; max-age=0;';
          resolve(null);
        });
      });
    },
    // logout of any current session and impersonate a user
    async impersonate(userToken: string, tokenExpires: string) {
      // set cookie with name THORIUM_TOKEN
      document.cookie = buildCookie(userToken, tokenExpires);
      setToken(userToken);
    },
  };
}

/**
 * Wrap application in a shared auth provider
 */
export const AuthProvider: React.FC<AuthHookProps> = ({ children }) => {
  const auth = useAuthProvider();
  return <authContext.Provider value={auth}>{children}</authContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(authContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within a AuthProvider');
  }
  return context;
};

type AuthHookProps = {
  children: JSX.Element;
};

/**
 * Validate that user is logged and redirect on validation failure
 */
export const RequireAuth: React.FC<AuthHookProps> = ({ children }) => {
  const { token } = useAuth();
  // token must be set and cookie must still be set
  // cookie gets cleared when it is expired

  return token != undefined && getCookie('THORIUM_TOKEN') != undefined ? (
    children
  ) : (
    <Navigate to="/auth" replace state={{ path: location.pathname + location.search + location.hash }} />
  );
};

/**
 * Validate user's Thorium role is admin and redirects on authorization failure
 */
export const RequireAdmin: React.FC<AuthHookProps> = ({ children }) => {
  const { userInfo } = useAuth();
  const role = userInfo?.role as unknown as RoleKey;
  return role == RoleKey.Admin ? (
    children
  ) : (
    <Navigate replace to={window.location.href} state={{ path: location.pathname + location.search + location.hash }} />
  );
};
