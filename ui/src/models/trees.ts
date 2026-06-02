// project imports
import { Association } from './associations';
import { Entities, EntityTypes } from './entities/entities';
import { Repo } from './entities/repos';
import { Origin, Sample } from './files';
import { TreeTags } from './tags';

/// The seed data to build a tree with
export interface Seed {
  /// Sample sha256s to start with
  samples?: string[];
  /// Device entity IDs to start with
  devices?: string[];
  /// Repo URLs to start with
  repos?: string[];
  /// Entity IDs to start with
  entities?: string[];
  /// Tags to start with
  tags?: {
    [key: string]: string[];
  };
  /// Related query settings
  related?: {
    tags?: {
      [key: string]: string[];
    };
  };
}

/// The direction for a relationship or branch
export enum Direction {
  /// From the parent to a child
  To = 'To',
  /// From the child to the parent
  From = 'From',
  /// Both directions
  Bidirectional = 'Bidirectional',
}

/// The type of relationship between tree nodes
export interface TreeRelationships {
  /// This is an initial node
  Initial?: 'Initial';
  /// This node is related by tags
  Tags?: 'Tags';
  /// This node is related by an association
  Association?: Association;
  /// This node is related due to an origin
  Origin?: Origin;
}

/// A branch between nodes in a relationship tree
export interface BranchNode {
  /// The relationship for this branch
  relationship: TreeRelationships;
  /// The node this is a branch to
  node: string;
  /// The direction for this branch
  direction: Direction;
  /// A hash for this relationship
  relationship_hash: string;
}

/// The types of nodes in a tree
export enum TreeNodeKey {
  /// A sample in Thorium
  Sample = 'Sample',
  /// A repo in Thorium
  Repo = 'Repo',
  /// A single specific tag in Thorium
  Tag = 'Tag',
  /// An entity in Thorium
  Entity = 'Entity',
}

/// A node in a tree
export type TreeNode = {
  [TreeNodeKey.Sample]?: Sample;
  [TreeNodeKey.Repo]?: Repo;
  [TreeNodeKey.Tag]?: TreeTags;
  [TreeNodeKey.Entity]?: EntityTypes;
};

/// The settings to use to relate in-tree nodes with other data
export interface TreeRelatedQuery {
  /// The tags to use when finding related data
  tags: { [key: string]: string[] }[];
}

/// A relationship tree
export interface Graph {
  /// This tree's id
  id: string;
  /// The groups this tree will search
  groups?: string[];
  /// The initial nodes for this tree
  initial: string[];
  /// The nodes that can be grown more on this tree
  growable: string[];
  /// The info on each node in this tree
  data_map: {
    [nodeId: string]: TreeNode;
  };
  /// The data in the leaves of this tree
  branches: {
    [nodeId: string]: BranchNode[];
  };
  /// The branches to hint data
  hint_branches?: {
    [nodeId: string]: BranchNode[];
  };
  /// The settings to use to relate in-tree nodes with other data
  related?: TreeRelatedQuery;
  /// The nodes that have already been sent
  sent?: string[];
}

export const BlankGraph: Graph = {
  id: '',
  initial: [],
  growable: [],
  data_map: {},
  branches: {},
};

export enum GraphTag {
  Tag = 'Tag',
}
export type NodeType = Entities | GraphTag;
export const NodeType = { ...Entities, ...GraphTag };
