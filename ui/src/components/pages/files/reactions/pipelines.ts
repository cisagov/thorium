import { Pipeline } from '@models/pipelines';

// spec: ../files.spec.md

const orderComparePipelineName = (a: Pick<Pipeline, 'name'>, b: Pick<Pipeline, 'name'>): number => {
  return a.name.localeCompare(b.name);
};

const orderComparePipeline = (a: Pick<Pipeline, 'group' | 'name'>, b: Pick<Pipeline, 'group' | 'name'>): number => {
  return (a.group + a.name).localeCompare(b.group + b.name);
};

export { orderComparePipeline, orderComparePipelineName };
