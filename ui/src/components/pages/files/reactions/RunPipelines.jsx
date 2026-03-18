import { useState } from 'react';
import { Button, Row } from 'react-bootstrap';

// project imports
import RunReactionAlerts from './RunReactionAlerts';
import SelectPipelines from './SelectPipelines';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import { useAuth } from '@utilities/auth';

const RunPipelines = ({ sha256 }) => {
  const { userInfo } = useAuth();
  const [reactionsList, setReactionsList] = useState([]);
  const [runReactionResponses, setRunReactionResponses] = useState([]);
  const [running, setRunning] = useState(false);

  // handle the reaction submission and setting of responses
  // this must be wrapped in a function object because of the async call
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
          <Button className="ok-btn auto-width" onClick={() => handleSubmitReactions()}>
            Run Pipelines
          </Button>
        )}
      </Row>
    </div>
  );
};

export default RunPipelines;
