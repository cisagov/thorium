import { Link } from 'react-router';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';

// Alert component for error and info responses for component submission
const RunReactionAlerts = ({ responses }) => {
  return (
    <>
      {responses.length > 0 &&
        responses.map((runResponse, idx) => (
          <div className="my-1" key={idx}>
            {runResponse.error && <AlertBanner className="full-width">{runResponse.error}</AlertBanner>}
            {runResponse.error == '' && (
              <AlertBanner severity={Severity.Info} className="my-2 full-width">
                <span>
                  {`Successfully submitted reaction `}
                  <Link className="link-text" to={`/reaction/${runResponse.group}/${runResponse.id}`} target="_blank">
                    {runResponse.id}
                  </Link>
                  {` for pipeline ${runResponse.pipeline} from group ${runResponse.group}!`}
                </span>
              </AlertBanner>
            )}
          </div>
        ))}
    </>
  );
};

export default RunReactionAlerts;
