// project imports
import { Association } from './associations';
import { Entities, EntityTypes } from './entities/entities';
import { Repo } from './entities/repos';
import { Origin, Sample } from './files';
import { TreeTags } from './tags';

export interface Seed {
  samples?: string[];
  devices?: string[];
  repos?: string[];
  entities?: string[];
  tags?: {
    [key: string]: string[];
  };
  related?: {
    tags?: {
      [key: string]: string[];
    };
  };
}

export enum Direction {
  To = 'To',
  From = 'From',
  Bidirectional = 'Bidirectional',
}

export interface TreeRelationships {
  Initial?: 'Initial';
  Tags?: 'Tags';
  Association?: Association;
  Origin?: Origin;
}

export interface BranchNode {
  relationship: TreeRelationships;
  node: string;
  direction: Direction;
  relationship_hash: string;
}

export type TreeNode = {
  Sample?: Sample;
  Repo?: Repo;
  Tag?: TreeTags;
  Entity?: EntityTypes;
};

export interface Graph {
  id: string;
  initial: string[];
  growable: string[];
  data_map: {
    [nodeId: string]: TreeNode;
  };
  branches: {
    [nodeId: string]: BranchNode[];
  };
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
