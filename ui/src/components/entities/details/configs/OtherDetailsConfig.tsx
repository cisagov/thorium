import { JSX } from 'react';
import { FaGhost } from 'react-icons/fa';

// project imports
import { EntityDetailsConfig } from './configs';
import { DetailsMetadataProps } from '../EntityDetails';
import { getEntity } from '@thorpi/entities';
import { Entities } from '@models/entities';
import { BlankOther, Other } from '@models/entities/other';

const OtherMetaInfo = ({ entity, pendingEntity, handleUpdate, editing }: DetailsMetadataProps<Entities.Other>): JSX.Element => {
  return <></>;
};

// Get Other entity details from the API
const getOtherDetails = async (vendorID: string, setError: (err: string) => void, updateEntity: (entity: Other) => void) => {
  getEntity(vendorID, setError).then((data) => {
    // check data is not null and is of Other kind
    if (data && data.kind == Entities.Other) {
      // we know that any entity response with kind=Other is a Other
      const other = data as Other;
      updateEntity(other);
    }
  });
};

const OtherDetailsConfig: EntityDetailsConfig<Entities.Other> = {
  getEntityDetails: getOtherDetails,
  EntityMetaInfo: OtherMetaInfo,
  BlankEntity: BlankOther,
  icon: (size: number) => <FaGhost size={size} />,
};

export default OtherDetailsConfig;
