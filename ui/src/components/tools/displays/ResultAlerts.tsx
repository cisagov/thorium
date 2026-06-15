import React from 'react';

// project imports
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';

interface ResultAlertsProps {
  errors: string[];
  warnings: string[];
}

/**
 * Render the error/warning alert banners shared by the tool-result display components: errors as
 * default (error) banners followed by warnings as warning banners.
 */
const ResultAlerts: React.FC<ResultAlertsProps> = ({ errors, warnings }) => (
  <>
    {errors.map((err, idx) => (
      <AlertBanner key={idx}>{err}</AlertBanner>
    ))}
    {warnings.map((warn, idx) => (
      <AlertBanner key={idx} severity={Severity.Warning}>
        {warn}
      </AlertBanner>
    ))}
  </>
);

export default ResultAlerts;
