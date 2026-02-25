export enum RoleKey {
  Admin = 'Admin',
  Analyst = 'Analyst',
  Developer = 'Developer',
  User = 'User',
  Reporter = 'Reporter',
}

type ThoriumDeveloperRoleValue = { k8s: boolean; bare_metal: boolean; windows: boolean; external: boolean; kvm: boolean };

export type Role = {
  Admin: RoleKey.Admin;
  Analyst: RoleKey.Analyst;
  Developer: {
    Developer: ThoriumDeveloperRoleValue;
  };
  User: RoleKey.User;
  Reporter: RoleKey.Reporter;
};

export type ThoriumRole = {
  Admin?: RoleKey.Admin;
  Analyst?: RoleKey.Analyst;
  Developer?: ThoriumDeveloperRoleValue;
  User?: RoleKey.User;
  Reporter?: RoleKey.Reporter;
};

export type UserInfo = {
  username: string;
  role: ThoriumRole;
  email: string;
  groups: string[];
  token: string;
  token_expiration: string;
  settings: {
    theme: string;
  };
  local: boolean;
  verified: boolean;
};

export type UserAuthResponse = {
  token: string; // Thorium auth token
  expires: string; // expiration date
};
