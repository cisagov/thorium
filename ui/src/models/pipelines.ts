import { TagTypes } from './tags';

export enum EventTriggerType {
  Tag = 'Tag',
  NewSample = 'NewSample',
  NewRepo = 'NewRepo',
}

export type EventTrigger = {
  [type in EventTriggerType]: {
    tag_types: TagTypes[];
    required: {
      [tagKey: string]: string[]; // tag key and array of values
    };
    not: {
      [tagKey: string]: string[]; // tag key and array of values
    };
  };
};

/// Contains data related to a [`PipelineBanKind::Generic`]
export type GenericBan = {
  /// A message containing the reason this pipeline was banned
  msg: String;
};

/// Contains data related to a [`PipelineBanKind::BannedImage`]
export type BannedImageBan = {
  /// The image in the pipeline that is banned
  image: String;
};

export type PipelineBanKind = {
  /// A generic ban manually set by an admin
  Generic: GenericBan;
  /// Created when an image in a pipeline has one or more bans
  BannedImage: BannedImageBan;
};

export type PipelineBan = {
  /// The unique id for this ban
  typeid: string;
  /// The time in UTC that the ban was made
  time_banned: string;
  /// The kind of ban this is
  typeban_kind: PipelineBanKind;
};

/// A particular reason an image has been banned
export type Pipeline = {
  /// The group this pipeline is tied to
  group: string;
  /// The name of this pipeline
  name: string;
  /// The creator of this pipeline
  creator: string;
  /// The order of images to be executed in this pipeline
  order: [string[]];
  /// The number of seconds we have to meet this pipelines SLA.
  sla: number;
  /// The triggers to execute this pipeline on
  triggers?: { [name: string]: EventTrigger };
  /// The description of the pipeline
  description?: string;
  /// A list of reasons the pipeline is banned mapped by ban UUID;
  /// if the list has any bans, the pipeline cannot be run
  bans?: { [uuid: string]: PipelineBan };
};

export type PipelineCreate = Omit<Pipeline, 'creator' | 'bans'>;

export type PipelineUpdate = Omit<PipelineCreate, 'name' | 'group'> & {
  remove_triggers?: string[]; // array of trigger names to remove
  clear_description?: boolean; // whether to clear the description
  bans?: { [uuid: string]: PipelineBan };
};
