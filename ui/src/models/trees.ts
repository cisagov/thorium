import { Association } from './associations';
import { Entity } from './entities/entities';
import { Repo } from './repos';
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
  Entity?: Entity;
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

export enum NodeType {
  File = 'File',
  Repo = 'Repo',
  Tag = 'Tag',
  Device = 'Device',
  Vendor = 'Vendor',
  Other = 'Other',
}
