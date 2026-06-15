// spec: ./SPEC.md

import React, { useCallback, useEffect, useMemo, useState } from 'react';

// project imports
import { PickerCard, PickerHeading, PickerIntro, PickerSelectRow } from './styles';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';
import LoadingSpinner from '@components/shared/fallback/LoadingSpinner';
import SelectInput from '@components/shared/inputs/selectable/SelectInput';
import { listEntities } from '@thorpi/entities';
import { Entities } from '@models/entities/entities';
import type { Incident } from '@models/entities/incident';
import type { ValueMap } from '@models/shared';

/// Props for {@link IncidentPicker}.
export interface IncidentPickerProps {
  /// Called with the chosen incident id when the user selects one from the list.
  onSelect: (incidentId: string) => void;
}

/// The state of the incident list fetch, exposed so the picker can render loading/error/empty distinctly.
interface IncidentListState {
  /// The fetched incidents (id + name), empty until the request resolves.
  incidents: Incident[];
  /// Whether the initial fetch is still in flight.
  loading: boolean;
  /// The fetch error message, or `null` when there is none.
  error: string | null;
}

/**
 * Fetch the incidents available to the current user for the picker.
 *
 * Requests full entity objects (`details = true`) filtered to {@link Entities.Incident} so each option
 * carries a human-readable `name`; a single page is sufficient for the picker (no cursor paging).
 *
 * @returns The incident list plus its loading/error state, refreshed once on mount.
 */
function useIncidentList(): IncidentListState {
  const [state, setState] = useState<IncidentListState>({ incidents: [], loading: true, error: null });
  useEffect(() => {
    // guard against a state update after unmount while the request is in flight
    let active = true;
    let requestError: string | null = null;
    const errorHandler = (message: string) => {
      requestError = message;
    };
    void listEntities({ kinds: [Entities.Incident] }, errorHandler, true, null).then(({ entityList }) => {
      if (!active) {
        return;
      }
      // listEntities resolves with an empty list on failure; surface the captured error message in that case
      setState({ incidents: entityList as Incident[], loading: false, error: requestError });
    });
    return () => {
      active = false;
    };
  }, []);
  return state;
}

/**
 * The incident picker shown when no `?incident` param is present on the incident dashboard.
 *
 * Renders a dismissible info tip and a non-creatable {@link SelectInput} whose options are the ids of
 * every {@link Entities.Incident} entity, labelled by `name` (falling back to the id when the name is
 * blank). Selecting an option calls {@link IncidentPickerProps.onSelect} with the chosen id, which the
 * page uses to set the `?incident=<id>` URL param. Loading, empty, and error states are handled inline.
 *
 * @param props - The picker props (see {@link IncidentPickerProps}).
 * @returns The picker UI.
 */
const IncidentPicker: React.FC<IncidentPickerProps> = ({ onSelect }) => {
  const { incidents, loading, error } = useIncidentList();

  // the select works in ids; the valueMap turns each id into its display name (or the id when name is blank)
  const optionIds = useMemo(() => incidents.map((incident) => incident.id), [incidents]);
  const valueMap = useMemo<ValueMap>(() => {
    const map: ValueMap = {};
    for (const incident of incidents) {
      map[incident.id] = incident.name && incident.name.trim() !== '' ? incident.name : incident.id;
    }
    return map;
  }, [incidents]);

  const handleChange = useCallback(
    (id: string) => {
      // ignore the clear action (empty string); only a real selection should navigate
      if (id !== '') {
        onSelect(id);
      }
    },
    [onSelect],
  );

  return (
    <PickerCard>
      <PickerHeading>Incident Dashboard</PickerHeading>
      <AlertBanner severity={Severity.Info} dismissible>
        Select an incident to view its dashboard.
      </AlertBanner>
      <PickerIntro>Choose an incident to open a dashboard seeded with its associated files, entities, and repos.</PickerIntro>
      {loading ? (
        <LoadingSpinner loading />
      ) : error ? (
        <AlertBanner severity={Severity.Error}>{error}</AlertBanner>
      ) : incidents.length === 0 ? (
        <AlertBanner severity={Severity.Warning}>No incidents found.</AlertBanner>
      ) : (
        <PickerSelectRow>
          <SelectInput
            options={optionIds}
            valueMap={valueMap}
            onChange={handleChange}
            disabled={false}
            defaultMessage="Select an incident..."
          />
        </PickerSelectRow>
      )}
    </PickerCard>
  );
};

export default IncidentPicker;
