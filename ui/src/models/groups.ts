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
