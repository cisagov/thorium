import React from 'react';
import { useUpload } from './UploadContext';
import OriginField from './OriginField';

const OriginIncident: React.FC = () => {
  const { originState, origin } = useUpload();
  const { incident, coverTerm, missionTeam, network, machine, location } = originState.incident;

  return (
    <>
      <OriginField
        label="Incident ID"
        value={incident}
        onChange={(v) => origin.setIncidentField('incident', v)}
        placeholder="name"
        isInvalid={!incident && (!!coverTerm || !!missionTeam || !!network || !!machine || !!location)}
        feedback="Please enter an Incident ID."
      />
      <OriginField label="Cover Term" value={coverTerm} onChange={(v) => origin.setIncidentField('coverTerm', v)} />
      <OriginField label="Mission Team" value={missionTeam} onChange={(v) => origin.setIncidentField('missionTeam', v)} />
      <OriginField label="Network" value={network} onChange={(v) => origin.setIncidentField('network', v)} />
      <OriginField label="Machine" value={machine} onChange={(v) => origin.setIncidentField('machine', v)} />
      <OriginField label="Location" value={location} onChange={(v) => origin.setIncidentField('location', v)} />
    </>
  );
};

export default OriginIncident;
