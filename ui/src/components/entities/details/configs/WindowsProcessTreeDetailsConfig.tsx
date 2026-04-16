import { JSX } from 'react';
import { Row } from 'react-bootstrap';
import { MdOutlineAccountTree } from 'react-icons/md';

// project imports
import { EntityDetailsConfig } from './configs';
import { DetailsMetadataProps } from '../EntityDetails';
import InfoValue from '../../shared/InfoValue';
import InfoHeader from '../../shared/InfoHeader';
import FieldBadge from '@components/shared/badges/FieldBadge';
import SelectInputArray from '@components/shared/inputs/selectable/SelectInputArray';
import { getEntity } from '@thorpi/entities';
import { Entities } from '@models/entities';
import { BlankWindowsProcessTree, WindowsProcessTree, WindowsProcessTreeCreateMetaFields } from '@models/entities/process_trees';

const ProcessTreeMetaInfo = ({
  entity,
  pendingEntity,
  handleUpdate,
  editing,
}: DetailsMetadataProps<Entities.WindowsProcessTree>): JSX.Element => {
  // update metadata and then pass back to entity update
  function updatePendingMeta<T extends keyof WindowsProcessTreeCreateMetaFields>(
    field: T,
    value: WindowsProcessTreeCreateMetaFields[T],
  ): void {
    const updates: WindowsProcessTreeCreateMetaFields = structuredClone(pendingEntity.metadata.WindowsProcessTree);
    updates[field] = value;
    handleUpdate('metadata', { WindowsProcessTree: updates });
  }
  return (
    <>
      <Row className="mt-3">
        <InfoHeader>Tools</InfoHeader>
        <InfoValue>
          {editing ? (
            <SelectInputArray
              isCreatable={true}
              values={pendingEntity.metadata.WindowsProcessTree.tools?.length > 0 ? pendingEntity.metadata.WindowsProcessTree.tools : []}
              options={pendingEntity.metadata.WindowsProcessTree.tools}
              onChange={(tools) => updatePendingMeta('tools', tools)}
              valuesMap={{}}
            />
          ) : (
            entity.metadata.WindowsProcessTree.tools.map((tool: string) => (
              <FieldBadge color="Gray" noNull={true} field={tool}></FieldBadge>
            ))
          )}
        </InfoValue>
      </Row>
    </>
  );
};

// Get process tree details from the API
const getProcessTreeDetails = async (
  vendorID: string,
  setError: (err: string) => void,
  updateEntity: (entity: WindowsProcessTree) => void,
) => {
  getEntity(vendorID, setError).then((data) => {
    // check data is not null and is of ProcessTree kind
    if (data && data.kind == Entities.WindowsProcessTree) {
      // we know that any entity response with kind=ProcessTree is a ProcessTree
      const tree = data as WindowsProcessTree;
      updateEntity(tree);
    }
  });
};

const WindowsProcessTreeDetailsConfig: EntityDetailsConfig<Entities.WindowsProcessTree> = {
  getEntityDetails: getProcessTreeDetails,
  EntityMetaInfo: ProcessTreeMetaInfo,
  BlankEntity: BlankWindowsProcessTree,
  icon: (size: number) => <MdOutlineAccountTree size={size} />,
};

export default WindowsProcessTreeDetailsConfig;
