import { JSX } from 'react';
import { Row } from 'react-bootstrap';
import { FaFolderTree } from 'react-icons/fa6';

// project imports
import InfoValue from '@entities/shared/InfoValue';
import InfoHeader from '@entities/shared/InfoHeader';
import EntityDetails from '@entities/details/EntityDetails';
import FieldBadge from '@components/shared/badges/FieldBadge';
import SelectInputArray from '@components/shared/selectable/SelectInputArray';
import { BlankFileSystem, FileSystem, FileSystemMetaFields } from '@models/entities/file_systems';
import { Entities, Entity } from '@models/entities';
import { getEntity } from '@thorpi/entities';

const FileSystemMetaInfo = (
  filesystem: FileSystem,
  pendingFileSystem: FileSystem,
  handleUpdate: <K extends keyof Entity>(field: K, value: Entity[K]) => void,
  editing: boolean,
): JSX.Element => {
  // update metadata and then pass back to entity update
  function updatePendingMeta<T extends keyof FileSystemMetaFields>(field: T, value: FileSystemMetaFields[T]): void {
    const updates: FileSystemMetaFields = structuredClone(pendingFileSystem.metadata.FileSystem);
    updates[field] = value;
    handleUpdate('metadata', { FileSystem: updates });
  }

  return (
    <>
      <Row>
        <InfoHeader>SHA256</InfoHeader>
        <InfoValue>{filesystem.metadata.FileSystem.sha256}</InfoValue>
      </Row>
      <hr className="my-3" />
      <Row className="mt-3">
        <InfoHeader>Tools</InfoHeader>
        <InfoValue>
          {editing ? (
            <SelectInputArray
              isCreatable={true}
              values={filesystem.metadata.FileSystem.tools?.length > 0 ? filesystem.metadata.FileSystem.tools : []}
              options={filesystem.metadata.FileSystem.tools}
              onChange={(tools) => updatePendingMeta('tools', [tools] as any)}
              valuesMap={{}}
            />
          ) : (
            filesystem.metadata.FileSystem.tools.map((tool: string) => <FieldBadge color="Gray" noNull={true} field={tool}></FieldBadge>)
          )}
        </InfoValue>
      </Row>
    </>
  );
};

const FileSystemDetails = () => {
  // Get device details from the API
  const getFileSystemDetails = async (vendorID: string, setError: (err: string) => void, updateEntity: (entity: FileSystem) => void) => {
    getEntity(vendorID, setError).then((data) => {
      // check data is not null and is of FileSystem kind
      if (data && data.kind == Entities.FileSystem) {
        // we know that any entity response with kind=FileSystem is a FileSystem
        const device = data as FileSystem;
        // need to add spaces for more human friendly suggestions for Critical Sectors
        if (device.metadata.FileSystem?.critical_sectors) {
          device.metadata.FileSystem.critical_sectors = device.metadata.FileSystem.critical_sectors.map((sector: string) =>
            sector.replace(/([A-Z])/g, ' $1').trim(),
          );
        }
        updateEntity(device);
      }
    });
  };
  return (
    <EntityDetails
      getEntityDetails={getFileSystemDetails}
      metadata={FileSystemMetaInfo}
      blank={BlankFileSystem}
      icon={(size: number) => <FaFolderTree size={size} />}
    />
  );
};

export default FileSystemDetails;
