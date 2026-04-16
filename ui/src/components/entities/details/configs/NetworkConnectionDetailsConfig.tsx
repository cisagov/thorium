import { JSX } from 'react';
import { Row } from 'react-bootstrap';
import { PiNetwork } from 'react-icons/pi';

// project imports
import { EntityDetailsConfig } from './configs';
import { DetailsMetadataProps } from '../EntityDetails';
import InfoValue from '../../shared/InfoValue';
import InfoHeader from '../../shared/InfoHeader';
import FieldBadge from '@components/shared/badges/FieldBadge';
import { getEntity } from '@thorpi/entities';
import { Entities } from '@models/entities';
import { BlankNetworkConnection, NetworkConnection } from '@models/entities/network_connections';

const NetworkConnectionMetaInfo = ({
  entity,
  pendingEntity,
  handleUpdate,
  editing,
}: DetailsMetadataProps<Entities.NetworkConnection>): JSX.Element => {
  return (
    <>
      <Row className="mt-3">
        <InfoHeader>Destination IP</InfoHeader>
        <InfoValue>{entity.metadata.NetworkConnection.destination}</InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Destination Port</InfoHeader>
        <InfoValue>
          {entity.metadata.NetworkConnection.destination_port != undefined ? entity.metadata.NetworkConnection.destination_port : ''}
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Source IP</InfoHeader>
        <InfoValue>{entity.metadata.NetworkConnection.source}</InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Source Port</InfoHeader>
        <InfoValue>
          {entity.metadata.NetworkConnection.source_port != undefined ? entity.metadata.NetworkConnection.source_port : ''}
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Protocol</InfoHeader>
        <InfoValue>
          {entity.metadata.NetworkConnection.protocol != undefined ? (
            <FieldBadge color="Gray" noNull={true} field={entity.metadata.NetworkConnection.protocol} />
          ) : (
            ''
          )}
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>State</InfoHeader>
        <InfoValue>{entity.metadata.NetworkConnection.state != undefined ? entity.metadata.NetworkConnection.state : ''}</InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>PID</InfoHeader>
        <InfoValue>{entity.metadata.NetworkConnection.pid != undefined ? entity.metadata.NetworkConnection.pid.toString() : ''}</InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Process</InfoHeader>
        <InfoValue>{entity.metadata.NetworkConnection.process != undefined ? entity.metadata.NetworkConnection.process : ''}</InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Create Time</InfoHeader>
        <InfoValue>
          {entity.metadata.NetworkConnection.create_time != undefined ? entity.metadata.NetworkConnection.create_time : ''}
        </InfoValue>
      </Row>
    </>
  );
};

// Get process tree details from the API
const getNetworkConnectionDetails = async (
  vendorID: string,
  setError: (err: string) => void,
  updateEntity: (entity: NetworkConnection) => void,
) => {
  getEntity(vendorID, setError).then((data) => {
    // check data is not null and is of Process kind
    if (data && data.kind == Entities.NetworkConnection) {
      // we know that any entity response with kind=Process is a Process
      const device = data as NetworkConnection;
      updateEntity(device);
    }
  });
};

const NetworkConnectionDetailsConfig: EntityDetailsConfig<Entities.NetworkConnection> = {
  getEntityDetails: getNetworkConnectionDetails,
  EntityMetaInfo: NetworkConnectionMetaInfo,
  BlankEntity: BlankNetworkConnection,
  icon: (size: number) => <PiNetwork size={size} />,
};

export default NetworkConnectionDetailsConfig;
