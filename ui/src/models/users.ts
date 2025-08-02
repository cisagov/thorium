export enum RoleKey {
  Admin = 'Admin',
  Analyst = 'Analyst',
  Developer = 'Developer',
  User = 'User',
  Reporter = 'Reporter',
}

export type Role = {
  Admin: RoleKey.Admin;
  Analyst: RoleKey.Analyst;
  Developer: {
    Developer: ThoriumDeveloperRoleValue;
  };
  User: RoleKey.User;
  Reporter: RoleKey.Reporter;
};

type ThoriumDeveloperRoleValue = { k8s: boolean; bare_metal: boolean; windows: boolean; external: boolean; kvm: boolean };
type ThoriumRoleValue = string | ThoriumDeveloperRoleValue;

export type ThoriumRole = {
  [role in RoleKey]: ThoriumRoleValue;
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
