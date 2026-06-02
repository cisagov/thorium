import { useState } from 'react';
import { Button, Row } from 'react-bootstrap';

// project imports
import RunReactionAlerts from './RunReactionAlerts';
import SelectPipelines from './SelectPipelines';
import { submitReactions } from './reactions';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import { useAuth } from '@utilities/auth';
import type { ReactionRunResult, ReactionSelection } from '@models/reactions';

interface RunPipelinesProps {
  sha256: string;
}

const RunPipelines = ({ sha256 }: RunPipelinesProps) => {
  const { userInfo } = useAuth();
  const [reactionsList, setReactionsList] = useState<ReactionSelection[]>([]);
  const [runReactionResponses, setRunReactionResponses] = useState<ReactionRunResult[]>([]);
  const [running, setRunning] = useState(false);

  // handle the reaction submission
  const handleSubmitReactions = async () => {
    setRunning(true);
    const runResponses = await submitReactions(sha256, reactionsList);
    setRunReactionResponses(runResponses);
    setRunning(false);
  };

  return (
    <div id="runpipelines-tab">
      <SelectPipelines userInfo={userInfo} setReactionsList={setReactionsList} sha256={sha256} />
      <RunReactionAlerts responses={runReactionResponses} />
      <Row className="d-flex justify-content-center mt-2">
        {running ? (
          <LoadingSpinner loading={running}></LoadingSpinner>
        ) : (
          <Button
            className="ok-btn auto-width"
            onClick={() => {
              void handleSubmitReactions();
            }}
          >
            Run Pipelines
          </Button>
        )}
      </Row>
    </div>
  );
};

export default RunPipelines;
