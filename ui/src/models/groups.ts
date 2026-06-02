/// The roles a user can have within a group (mirrors the backend `Roles` enum)
export enum GroupRoleKey {
  Owner = 'Owner',
  Manager = 'Manager',
  /// Analysts have global access to all groups in Thorium
  Analyst = 'Analyst',
  Monitor = 'Monitor',
  User = 'User',
}

export type GroupAllowed = {
  /// Whether files are allowed to be added to this group
  files: boolean;
  /// Whether repos are allowed to be added to this group
  repos: boolean;
  /// Whether tags are allowed to be added to this group
  tags: boolean;
  /// Whether images are allowed to be added to this group
  images: boolean;
  /// Whether pipelines are allowed to be added to this group
  pipelines: boolean;
  /// Whether reactions are allowed to be created in this group
  reactions: boolean;
  /// Whether results are allowed to be added to this group
  results: boolean;
  /// Whether comments are allowed to be added to this group
  comments: boolean;
  /// Whether entities are allowed to be added to this group
  entities: boolean;
};

export type GroupUsers = {
  /// The combined direct users and members of metagroups for this role
  combined: string[];
  /// The users that were directly added to this group
  direct: string[];
  /// The metagroups that should have this role
  metagroups: string[];
};

export type Group = {
  /// The name of group
  name: string;
  /// Owners of this group.
  owners: GroupUsers;
  /// Managers of this group
  managers: GroupUsers;
  /// All analysts in Thorium
  analysts: string[];
  /// Users of this group.
  users: GroupUsers;
  /// Reporters of this group.
  monitors: GroupUsers;
  /// Description of the group,
  description?: string;
  /// The data that is allowed to be added to this group
  allowed: GroupAllowed;
};

/// Add/remove operations for a single role in a group update
export type GroupRoleUpdate = {
  direct_add?: string[];
  direct_remove?: string[];
  metagroups_add?: string[];
  metagroups_remove?: string[];
};

/// The request body for updating a group's membership and metadata
export type GroupUpdate = {
  description?: string;
  clear_description?: boolean;
  owners?: GroupRoleUpdate;
  managers?: GroupRoleUpdate;
  users?: GroupRoleUpdate;
  monitors?: GroupRoleUpdate;
};

export type GroupUserCategory = 'owners' | 'managers' | 'users' | 'monitors';
