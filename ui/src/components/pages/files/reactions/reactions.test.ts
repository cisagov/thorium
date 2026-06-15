import { describe, it, expect } from 'vitest';

// project imports
import { buildReactionsList, SelectedPipelines } from './reactions';

describe('buildReactionsList', () => {
  it('returns an empty list when no pipelines are selected', () => {
    const selected: SelectedPipelines = { group1: { pipelineA: false } };
    expect(buildReactionsList(selected)).toEqual([]);
  });

  it('includes only pipelines toggled on', () => {
    const selected: SelectedPipelines = {
      group1: { pipelineA: true, pipelineB: false },
      group2: { pipelineC: true },
    };
    const result = buildReactionsList(selected);
    expect(result).toHaveLength(2);
    expect(result.map((r) => r.pipeline).sort()).toEqual(['pipelineA', 'pipelineC']);
  });

  it('carries the group and default args/sla for each selected pipeline', () => {
    const selected: SelectedPipelines = { group1: { pipelineA: true } };
    const result = buildReactionsList(selected);
    expect(result[0]).toMatchObject({
      pipeline: 'pipelineA',
      group: 'group1',
      args: {},
      sla: 30,
    });
  });

  it('omits tags when none are provided', () => {
    const selected: SelectedPipelines = { group1: { pipelineA: true } };
    const result = buildReactionsList(selected);
    expect(result[0].tags).toBeUndefined();
  });

  it('omits tags when the tags object is empty', () => {
    const selected: SelectedPipelines = { group1: { pipelineA: true } };
    const result = buildReactionsList(selected, {});
    expect(result[0].tags).toBeUndefined();
  });

  it('attaches tags to every selected pipeline when provided', () => {
    const selected: SelectedPipelines = {
      group1: { pipelineA: true },
      group2: { pipelineB: true },
    };
    const tags = { family: ['emotet'] };
    const result = buildReactionsList(selected, tags);
    expect(result).toHaveLength(2);
    for (const reaction of result) {
      expect(reaction.tags).toEqual(tags);
    }
  });
});
