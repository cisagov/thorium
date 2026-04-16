import { JSX } from 'react';
import { Row } from 'react-bootstrap';
import { FaFolderTree } from 'react-icons/fa6';

// project imports
import { EntityDetailsConfig } from './configs';
import { DetailsMetadataProps } from '../EntityDetails';
import InfoValue from '@entities/shared/InfoValue';
import InfoHeader from '@entities/shared/InfoHeader';
import FieldBadge from '@components/shared/badges/FieldBadge';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import { Entities } from '@models/entities';
import { getEntity } from '@thorpi/entities';
import { BlankFileSystem, FileSystem, FileSystemMetaFields } from '@models/entities/file_systems';

const FileSystemMetaInfo = ({ entity, pendingEntity, handleUpdate, editing }: DetailsMetadataProps<Entities.FileSystem>): JSX.Element => {
  // update metadata and then pass back to entity update
  function updatePendingMeta<T extends keyof FileSystemMetaFields>(field: T, value: FileSystemMetaFields[T]): void {
    const updates: FileSystemMetaFields = structuredClone(pendingEntity.metadata.FileSystem);
    updates[field] = value;
    handleUpdate('metadata', { FileSystem: updates });
  }
  return (
    <>
      <Row>
        <InfoHeader>SHA256</InfoHeader>
        <InfoValue>{entity.metadata.FileSystem.sha256}</InfoValue>
      </Row>
      <hr className="my-3" />
      <Row className="mt-3">
        <InfoHeader>Tools</InfoHeader>
        <InfoValue>
          {editing ? (
            <SelectInputArray
              isCreatable={true}
              values={pendingEntity.metadata.FileSystem.tools}
              options={entity.metadata.FileSystem.tools}
              onChange={(tools) => updatePendingMeta('tools', tools)}
            />
          ) : (
            entity.metadata.FileSystem.tools.map((tool: string) => <FieldBadge color="Gray" noNull={true} field={tool}></FieldBadge>)
          )}
        </InfoValue>
      </Row>
    </>
  );
};

// Get filesystem details from the API
const getFileSystemDetails = async (vendorID: string, setError: (err: string) => void, updateEntity: (entity: FileSystem) => void) => {
  getEntity(vendorID, setError).then((data) => {
    // check data is not null and is of FileSystem kind
    if (data && data.kind == Entities.FileSystem) {
      // we know that any entity response with kind=FileSystem is a FileSystem
      const fs = data as FileSystem;
      updateEntity(fs);
    }
  });
};

const FileSystemDetailsConfig: EntityDetailsConfig<Entities.FileSystem> = {
  getEntityDetails: getFileSystemDetails,
  EntityMetaInfo: FileSystemMetaInfo,
  BlankEntity: BlankFileSystem,
  icon: (size: number) => <FaFolderTree size={size} />,
};

export default FileSystemDetailsConfig;
