import { Badge } from 'react-bootstrap';
import { FaCircle, FaCheckCircle, FaDotCircle, FaTimesCircle, FaSpinner } from 'react-icons/fa';

// project imports
import { createReaction, deleteReaction } from '@thorpi/reactions';

// Get the colored badge based on the status of a given reaction/job
const getStatusBadge = (status) => {
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

// Get the colored badge based on the status of a given reaction/job
const getStatusIcon = (status) => {
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

// internal function to build a list of reactions from a pipelines details and a
// list of selected pipelines
const buildReactionsList = (selectedPipelines, tags) => {
  // build selected Jobs list
  const reactionList = [];
  Object.keys(selectedPipelines).map((group) => {
    Object.keys(selectedPipelines[group]).map((pipeline) => {
      if (selectedPipelines[group][pipeline]) {
        const body = {
          pipeline: pipeline,
          group: group,
          args: {},
          sla: 30,
        };
        if (tags != []) {
          body['tags'] = tags;
        }
        reactionList.push(body);
      }
    });
  });
  return reactionList;
};

// submit reactions for a sha256 for a partially build reaction list containing
// reaction info for the selected pipelines
const submitReactions = async (sha256, reactionList) => {
  const reactionRunResults = [];
  for (const reaction of reactionList) {
    reaction.samples = [sha256];

    // handle adding the error to the results object for rendering
    const handleReactionCreationFailure = (error) => {
      reactionRunResults.push({
        error: 'Failed to submit ' + reaction.pipeline + ' for ' + sha256 + ': ' + error,
        group: reaction.group,
        pipeline: reaction.pipeline,
      });
    };

    const res = await createReaction(reaction, handleReactionCreationFailure);
    if (res) {
      // return response including reaction uuid and pipeline/group
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
