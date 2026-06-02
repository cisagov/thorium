import { ReactNode } from 'react';
import { Badge } from 'react-bootstrap';
import { FaCircle, FaCheckCircle, FaDotCircle, FaTimesCircle, FaSpinner } from 'react-icons/fa';

// project imports
import { createReaction, deleteReaction } from '@thorpi/reactions';
import { ReactionRequest, ReactionRunResult, ReactionSelection } from '@models/reactions';

type SelectedPipelines = Record<string, Record<string, boolean>>;

// get the colored badge based on the status of a given reaction/job
const getStatusBadge = (status: string): ReactNode => {
  switch (status) {
    case 'Completed':
      return <Badge bg="success">Completed</Badge>;
    case 'Failed':
    case 'Errored':
      return <Badge bg="danger">Failed</Badge>;
    case 'Created':
      return <Badge bg="secondary">Created</Badge>;
    case 'Running':
      return <Badge bg="primary">Running</Badge>;
    default:
      return <Badge bg="secondary">{status}</Badge>;
  }
};

// get the colored icon based on the status of a given reaction/job
const getStatusIcon = (status: string): ReactNode => {
  switch (status) {
    case 'Completed':
      return <FaCheckCircle size={18} color="green" />;
    case 'Failed':
      return <FaTimesCircle size={18} color="red" />;
    case 'Created':
      return <FaDotCircle size={18} color="lightBlue" />;
    case 'Running':
      return <FaSpinner size={18} color="blue" />;
    default:
      return <FaCircle size={18} color="grey" />;
  }
};

// build a list of reactions from a pipelines details and selected pipelines
const buildReactionsList = (selectedPipelines: SelectedPipelines, tags?: Record<string, string[]>) => {
  const reactionList: Array<{
    pipeline: string;
    group: string;
    args: Record<string, never>;
    sla: number;
    tags?: Record<string, string[]>;
  }> = [];
  Object.keys(selectedPipelines).map((group) => {
    Object.keys(selectedPipelines[group]).map((pipeline) => {
      if (selectedPipelines[group][pipeline]) {
        const body: {
          pipeline: string;
          group: string;
          args: Record<string, never>;
          sla: number;
          tags?: Record<string, string[]>;
        } = {
          pipeline: pipeline,
          group: group,
          args: {},
          sla: 30,
        };
        if (tags && Object.keys(tags).length > 0) {
          body.tags = tags;
        }
        reactionList.push(body);
      }
    });
  });
  return reactionList;
};

// submit reactions for a sha256
const submitReactions = async (sha256: string, reactionList: ReactionSelection[]): Promise<ReactionRunResult[]> => {
  const reactionRunResults: ReactionRunResult[] = [];
  for (const reaction of reactionList) {
    const request = {
      ...reaction,
      samples: [sha256],
      tags: reaction.tags
        ? Object.entries(reaction.tags)
            .flat()
            .filter((v): v is string => typeof v === 'string')
        : [],
    } as ReactionRequest;

    const handleReactionCreationFailure = (error: string) => {
      reactionRunResults.push({
        error: 'Failed to submit ' + reaction.pipeline + ' for ' + sha256 + ': ' + error,
        group: reaction.group,
        pipeline: reaction.pipeline,
      });
    };

    const res = await createReaction(request, handleReactionCreationFailure);
    if (res) {
      reactionRunResults.push({
        id: res.id,
        error: '',
        group: reaction.group,
        pipeline: reaction.pipeline,
      });
    }
  }
  return reactionRunResults;
};

export { buildReactionsList, submitReactions, deleteReaction, getStatusBadge, getStatusIcon };
export type { SelectedPipelines };
