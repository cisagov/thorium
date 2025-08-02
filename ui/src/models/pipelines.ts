import { TagTypes } from './tags';

export enum EventTriggerType {
  Tag = 'Tag',
  NewSample = 'NewSample',
  NewRepo = 'NewRepo',
}

export type EventTriggers = {
  [name: string]: {
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
};

export type Pipeline = {
  name: string;
  group: string;
  creator: string;
  order: [string[]];
  sla: number;
  triggers: EventTriggers | null;
  description: string | null;
  bans: any;
};

export type PipelineCreate = Omit<Pipeline, 'creator' | 'bans'> & {
  sla?: number;
  triggers?: EventTriggers;
};

export type PipelineUpdate = Omit<PipelineCreate, 'name' | 'group'> & {
  remove_triggers?: string[]; // array of trigger names to remove
  clear_description?: boolean; // whether to clear the description
  bans?: any;
};
