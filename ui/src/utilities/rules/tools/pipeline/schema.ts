import { FieldValueType, type FieldSchema } from '../../types';

export const REQUIRED_PIPELINE_FIELDS = ['group', 'name', 'order'] as const;

export const KNOWN_PIPELINE_FIELDS = ['group', 'name', 'order', 'sla', 'triggers', 'description'] as const;

export const EVENT_TRIGGER_VALUES = ['NewSample', 'Tag'] as const;
export const TAG_TYPE_VALUES = ['Files', 'Repos', 'Entities'] as const;
export const KNOWN_TAG_TRIGGER_FIELDS = ['tag_types', 'required', 'not'] as const;

export const PIPELINE_SECTION_ORDER = ['Pipeline', 'Triggers', 'Unknown Fields'] as const;

const PIPELINE_CATEGORY_MAP: Record<string, string> = {
  triggers: 'Triggers',
  order: 'Pipeline',
};

export function pipelineFieldCategory(field: string): string {
  const root = field.split('.')[0];
  return PIPELINE_CATEGORY_MAP[root] ?? 'Pipeline';
}

export const TAG_TRIGGER_SCHEMA: FieldSchema = {
  type: FieldValueType.Object,
  typeName: 'EventTrigger',
  fields: {
    tag_types: {
      type: FieldValueType.StringArray,
      enumValues: TAG_TYPE_VALUES,
      placeholder: 'Files',
      description: 'Tag types to trigger on (Files, Repos, Entities)',
    },
    required: { type: FieldValueType.Object, placeholder: 'tag-key', description: 'Tags that must be present to trigger' },
    not: { type: FieldValueType.Object, placeholder: 'tag-key', description: 'Tags that must not be present to trigger' },
  },
};

export const TAG_TRIGGER_FULL_SCHEMA: FieldSchema = {
  type: FieldValueType.Object,
  typeName: 'EventTrigger',
  fields: {
    Tag: TAG_TRIGGER_SCHEMA,
  },
};

export const PIPELINE_FIELD_SCHEMAS: Record<string, FieldSchema> = {
  group: { type: FieldValueType.String, required: true, placeholder: 'group-name', description: 'Group this pipeline belongs to' },
  name: { type: FieldValueType.String, required: true, placeholder: 'pipeline-name', description: 'Name of this pipeline' },
  description: { type: FieldValueType.String, placeholder: 'Pipeline description', description: 'Human-readable description' },
  sla: { type: FieldValueType.Number, placeholder: '604800', description: 'SLA deadline in seconds' },
  order: {
    type: FieldValueType.StringArray,
    nestedList: true,
    placeholder: 'image-name',
    description: 'Image execution order; each stage runs its images in parallel',
  },
  triggers: {
    type: FieldValueType.Object,
    fields: {
      'trigger-name': {
        type: FieldValueType.Enum,
        typeName: 'EventTrigger',
        enumValues: EVENT_TRIGGER_VALUES,
        variants: {
          NewSample: null,
          Tag: TAG_TRIGGER_SCHEMA,
        },
      },
    },
  },
};
