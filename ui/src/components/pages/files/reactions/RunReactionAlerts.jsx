import { Alert } from 'react-bootstrap';
import { Link } from 'react-router';

// Alert component for error and info responses for component submission
const RunReactionAlerts = ({ responses }) => {
  return (
    <>
      {responses.length > 0 &&
        responses.map((runResponse, idx) => (
          <div className="my-1" key={idx}>
            {runResponse.error && (
              <Alert className="full-width" variant="danger">
                <center>{runResponse.error}</center>
              </Alert>
            )}
            {runResponse.error == '' && (
              <Alert className="my-2 full-width" variant="info">
                <center>
                  <span>
                    {`Successfully submitted reaction `}
                    <Link className="link-text" to={`/reaction/${runResponse.group}/${runResponse.id}`} target="_blank">
                      {runResponse.id}
                    </Link>
                    {` for pipeline ${runResponse.pipeline} from group ${runResponse.group}!`}
                  </span>
                </center>
              </Alert>
            )}
          </div>
        ))}
    </>
  );
};

export default RunReactionAlerts;
