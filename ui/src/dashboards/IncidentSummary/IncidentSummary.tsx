// spec: ./SPEC.md

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';

// project imports
import IncidentPicker from './IncidentPicker';
import { ChangeIncidentButton, IncidentHeader, IncidentTitle } from './styles';
import { DashboardContent } from '../Dashboard';
import Page from '@components/pages/Page';
import { OverlayTipBottom } from '@components/shared/overlay/tips';
import { getEntity } from '@thorpi/entities';
import type { Seed } from '@models/trees';

/// The URL query key carrying the selected incident's entity id.
const INCIDENT_PARAM = 'incident';
/// The crawl depth for an incident dashboard, matching the general dashboard default.
const INCIDENT_DEPTH = 2;

/// Props for {@link IncidentDashboard}.
interface IncidentDashboardProps {
  /// The selected incident's entity id.
  incidentId: string;
  /// Clears the `?incident` param to return to the picker.
  onChangeIncident: () => void;
}

/**
 * The incident dashboard for a selected incident.
 *
 * Seeds the shared {@link DashboardContent} with the incident entity (memoized on the id so unrelated
 * URL edits never refetch the graph) and shows a header with the incident's resolved name plus a
 * "Change incident" affordance that returns to the picker.
 *
 * @param props - The dashboard props (see {@link IncidentDashboardProps}).
 * @returns The seeded dashboard.
 */
const IncidentDashboard: React.FC<IncidentDashboardProps> = ({ incidentId, onChangeIncident }) => {
  // the seed is a single entity (the incident); memoized on the id so DashboardContent's graph provider
  // is not remounted/refetched on every unrelated URL change (omnibar clauses, tab hash)
  const seed = useMemo<Seed>(() => ({ entities: [incidentId] }), [incidentId]);
  const [name, setName] = useState<string | null>(null);
  useEffect(() => {
    // resolve the incident's display name for the header; the graph seeding does not depend on this
    let active = true;
    setName(null);
    void getEntity(incidentId, () => {}).then((entity) => {
      if (active && entity && entity.name.trim() !== '') {
        setName(entity.name);
      }
    });
    return () => {
      active = false;
    };
  }, [incidentId]);

  return (
    <>
      <IncidentHeader>
        <IncidentTitle>Incident: {name ?? incidentId}</IncidentTitle>
        <OverlayTipBottom tip="Return to the incident picker to choose a different incident">
          <ChangeIncidentButton type="button" onClick={onChangeIncident}>
            Change incident
          </ChangeIncidentButton>
        </OverlayTipBottom>
      </IncidentHeader>
      <DashboardContent seed={seed} depthAtMount={INCIDENT_DEPTH} />
    </>
  );
};

/**
 * The incident dashboard page (`/dashboard/incident`).
 *
 * Reads `?incident=<uuid>` from the URL: when present it renders the seeded {@link IncidentDashboard};
 * when absent it renders the {@link IncidentPicker}, whose selection sets the `?incident` param. All
 * incident state lives in the URL so a dashboard is fully shareable/deep-linkable.
 *
 * @returns The incident dashboard page.
 */
const IncidentSummary: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const incidentId = searchParams.get(INCIDENT_PARAM);

  // set the incident param (replace so the picker->dashboard step isn't a separate back-button stop)
  const selectIncident = useCallback(
    (id: string) => {
      const next = new URLSearchParams(searchParams);
      next.set(INCIDENT_PARAM, id);
      setSearchParams(next, { replace: true });
    },
    [searchParams, setSearchParams],
  );

  // clear the incident param (and any leftover dashboard state) to return to the picker
  const clearIncident = useCallback(() => {
    setSearchParams(new URLSearchParams());
  }, [setSearchParams]);

  return (
    <Page title="Incident Dashboard" className="full-min-width">
      {incidentId ? (
        <IncidentDashboard incidentId={incidentId} onChangeIncident={clearIncident} />
      ) : (
        <IncidentPicker onSelect={selectIncident} />
      )}
    </Page>
  );
};

export default IncidentSummary;
