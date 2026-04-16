import { Row } from 'react-bootstrap';
import { JSX } from 'react';
import { FaFolder } from 'react-icons/fa';

// project imports
import { EntityDetailsConfig } from './configs';
import { DetailsMetadataProps } from '../EntityDetails';
import InfoValue from '@components/entities/shared/InfoValue';
import InfoHeader from '@components/entities/shared/InfoHeader';
import { getEntity } from '@thorpi/entities';
import { Entities } from '@models/entities/entities';
import { BlankFolder, Folder } from '@models/entities/folders';

// Get filesystem details from the API
const getFolderDetails = async (vendorID: string, setError: (err: string) => void, updateEntity: (entity: Folder) => void) => {
  getEntity(vendorID, setError).then((data) => {
    // check data is not null and is of Folder kind
    if (data && data.kind == Entities.Folder) {
      // we know that any entity response with kind=Folder is a Folder
      const folder = data as Folder;
      updateEntity(folder);
    }
  });
};

const FolderMetaInfo = ({ entity, pendingEntity, handleUpdate, editing }: DetailsMetadataProps<Entities.Folder>): JSX.Element => {
  return (
    <>
      <Row>
        <InfoHeader>File System ID</InfoHeader>
        <InfoValue>{entity.metadata.Folder.filesystem_id}</InfoValue>
      </Row>
      <hr className="my-3" />
      <Row className="mt-3">
        <InfoHeader>Names SHA256</InfoHeader>
        <InfoValue>{entity.metadata.Folder.names_sha256}</InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>Data SHA256</InfoHeader>
        <InfoValue>{entity.metadata.Folder.data_sha256}</InfoValue>
      </Row>
      <Row className="mt-3">
        <InfoHeader>All SHA256</InfoHeader>
        <InfoValue>{entity.metadata.Folder.all_sha256}</InfoValue>
      </Row>
    </>
  );
};

const FolderDetailsConfig: EntityDetailsConfig<Entities.Folder> = {
  getEntityDetails: getFolderDetails,
  EntityMetaInfo: FolderMetaInfo,
  BlankEntity: BlankFolder,
  icon: (size: number) => <FaFolder size={size} />,
};

export default FolderDetailsConfig;
