/// The roles a user can have (mirrors the backend `UserRole` enum)
export enum RoleKey {
  /// An admin can see all data in Thorium and perform any action
  Admin = 'Admin',
  /// An Analyst is given access to all samples in Thorium
  Analyst = 'Analyst',
  /// A developer can create tools
  Developer = 'Developer',
  /// A user can upload files and run jobs
  User = 'User',
}

/// The developer role's sub-permissions
type ThoriumDeveloperRoleValue = { k8s: boolean; bare_metal: boolean; windows: boolean; external: boolean; kvm: boolean };

export type Role = {
  Admin: RoleKey.Admin;
  Analyst: RoleKey.Analyst;
  Developer: {
    Developer: ThoriumDeveloperRoleValue;
  };
  User: RoleKey.User;
};

export type ThoriumRole = {
  Admin?: RoleKey.Admin;
  Analyst?: RoleKey.Analyst;
  Developer?: ThoriumDeveloperRoleValue;
  User?: RoleKey.User;
};

/// The info to inject about this user on a Unix/Linux system
export type UnixInfo = {
  /// The unix user id of this user
  user: number;
  /// The unix group id of this user
  group: number;
};

/// An AI endpoint configuration
export type AiEndpoint = {
  /// The URL endpoint for the AI service
  url: string;
  /// The API key for the AI service
  api_key: string;
  /// The model to use from this AI service
  model: string;
};

/// AI settings for a user
export type AiSettings = {
  /// Multiple named AI endpoints
  endpoints: { [name: string]: AiEndpoint };
  /// The default endpoint name
  default_endpoint: string;
};

/// Any user specific settings
export type UserSettings = {
  /// The theme this user uses in the webUI
  theme: string;
  /// The AI settings for this user
  ai?: AiSettings;
};

/// A user within Thorium that does not have its password
export type UserInfo = {
  /// The username of this user
  username: string;
  /// The role for this user
  role: ThoriumRole;
  /// This users email
  email: string;
  /// The groups this user is in
  groups: string[];
  /// The token for this user
  token: string;
  /// When this users token expires
  token_expiration: string;
  /// The info to inject about this user on unix/linux systems
  unix?: UnixInfo;
  /// The settings this user has set
  settings: UserSettings;
  /// Whether this user is a local user or not
  local: boolean;
  /// Whether this user has been verified already or not
  verified: boolean;
};

/// Response to a successful auth
export type UserAuthResponse = {
  /// The token to use to talk to Thorium
  token: string;
  /// The date/time this token expires
  expires: string;
};
