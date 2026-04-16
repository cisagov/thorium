import { JSX } from 'react';
import { HiOutlineArrowTurnDownRight } from 'react-icons/hi2';
import { Row } from 'react-bootstrap';

// project imports
import { EntityDetailsConfig } from './configs';
import { DetailsMetadataProps } from '../EntityDetails';
import InfoValue from '../../shared/InfoValue';
import InfoHeader from '../../shared/InfoHeader';
import FieldBadge from '@components/shared/badges/FieldBadge';
import { getEntity } from '@thorpi/entities';
import { Entities } from '@models/entities';
import { BlankWindowsProcess, WindowsProcess } from '@models/entities/processes';

const WindowsProcessMetaInfo = ({
  entity,
  pendingEntity,
  handleUpdate,
  editing,
}: DetailsMetadataProps<Entities.WindowsProcess>): JSX.Element => {
  return (
    <>
      <Row>
        <InfoHeader>Name</InfoHeader>
        <InfoValue>{entity.metadata.WindowsProcess.name ? entity.metadata.WindowsProcess.name : entity.name}</InfoValue>
      </Row>
      <hr className="my-3" />
      <Row className="mt-3">
        <InfoHeader>Command</InfoHeader>
        <InfoValue>{entity.metadata.WindowsProcess.command != undefined ? entity.metadata.WindowsProcess.command : ''}</InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Image Path</InfoHeader>
        <InfoValue>{entity.metadata.WindowsProcess.image_path != undefined ? entity.metadata.WindowsProcess.image_path : ''}</InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>PID</InfoHeader>
        <InfoValue>{<FieldBadge color="Gray" noNull={true} field={entity.metadata.WindowsProcess.pid} />}</InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Parent PID</InfoHeader>
        <InfoValue>
          {entity.metadata.WindowsProcess.parent_pid != undefined ? (
            <FieldBadge color="Gray" noNull={true} field={entity.metadata.WindowsProcess.parent_pid} />
          ) : (
            ''
          )}
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Offset</InfoHeader>
        <InfoValue>
          {entity.metadata.WindowsProcess.offset != undefined ? `0x${entity.metadata.WindowsProcess.offset.toString(16)}` : ''}
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Threads</InfoHeader>
        <InfoValue>{entity.metadata.WindowsProcess.threads ? entity.metadata.WindowsProcess.threads : ''}</InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>WOW64</InfoHeader>
        <InfoValue>
          <FieldBadge color="DarkRed" noNull={true} field={entity.metadata.WindowsProcess.is_wow64} />
        </InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Session ID</InfoHeader>
        <InfoValue>{entity.metadata.WindowsProcess.session_id != undefined ? entity.metadata.WindowsProcess.session_id : ''}</InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Create Time</InfoHeader>
        <InfoValue>{entity.metadata.WindowsProcess.create_time != undefined ? entity.metadata.WindowsProcess.create_time : ''}</InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Exit Time</InfoHeader>
        <InfoValue>{entity.metadata.WindowsProcess.exit_time != undefined ? entity.metadata.WindowsProcess.exit_time : ''}</InfoValue>
      </Row>
    </>
  );
};

// Get process tree details from the API
const getWindowsProcessDetails = async (
  vendorID: string,
  setError: (err: string) => void,
  updateEntity: (entity: WindowsProcess) => void,
) => {
  getEntity(vendorID, setError).then((data) => {
    // check data is not null and is of Process kind
    if (data && data.kind == Entities.WindowsProcess) {
      // we know that any entity response with kind=Process is a Process
      const device = data as WindowsProcess;
      updateEntity(device);
    }
  });
};

const WindowsProcessDetailsConfig: EntityDetailsConfig<Entities.WindowsProcess> = {
  getEntityDetails: getWindowsProcessDetails,
  EntityMetaInfo: WindowsProcessMetaInfo,
  BlankEntity: BlankWindowsProcess,
  icon: (size: number) => <HiOutlineArrowTurnDownRight size={size} />,
};

export default WindowsProcessDetailsConfig;
