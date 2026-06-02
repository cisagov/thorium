import { Link } from 'react-router';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';

// project imports
import { ReactionRunResult } from '@models/reactions';

interface RunReactionAlertsProps {
  responses: ReactionRunResult[];
}

const RunReactionAlerts = ({ responses }: RunReactionAlertsProps) => {
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
