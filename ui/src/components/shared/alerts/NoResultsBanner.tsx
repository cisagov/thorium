// project imports
import AlertBanner, { Severity } from './AlertBanner';

interface NoResultsBannerProps {
  type?: string; // resource label, e.g. "Images"; omit for a generic "None Found"
  className?: string; // defaults to 'm-1' to match the entity browsing list
}

/**
 * Info banner shown when a list or filtered search returns no results.
 *
 * @param type - The resource label to interpolate ("No {type} Found"); when omitted, renders "None Found".
 * @param className - Optional wrapper class; defaults to `'m-1'`.
 * @returns An info-severity {@link AlertBanner} with the no-results message.
 */
const NoResultsBanner: React.FC<NoResultsBannerProps> = ({ type, className = 'm-1' }) => (
  <AlertBanner severity={Severity.Info} className={className}>
    {type ? <>No {type} Found</> : <>None Found</>}
  </AlertBanner>
);

export default NoResultsBanner;
