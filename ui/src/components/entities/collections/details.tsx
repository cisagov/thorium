import { Button } from 'react-bootstrap';
import { FaMagnifyingGlass } from 'react-icons/fa6';
import { useNavigate } from 'react-router-dom';

import { OverlayTipBottom } from '@components';
import { Collection, CollectionMeta, Entity } from 'models';

type ListCollectionProps = {
  entity: Entity;
};

export const ListCollectionButton: React.FC<ListCollectionProps> = ({ entity }) => {
  const navigate = useNavigate();

  const buildFilesUrl = (collection: Collection): string => {
    const params = new URLSearchParams();
    const metadata: CollectionMeta = collection.metadata;
    if (!metadata.Collection.ignore_groups) {
      (collection.groups ?? []).forEach((g) => params.append('groups', g));
    }
    Object.entries(metadata.Collection.collection_tags ?? {}).forEach(([key, values]) => {
      values.forEach((v) => {
        const encodedKey = `tags[${key}]`; // e.g. tags[a]
        params.append(encodedKey, v);
      });
    });
    const { start, end } = metadata.Collection;
    if (start) {
      params.set('start', start);
    }
    if (end) {
      params.set('end', end);
    }
    params.set('tags_case_insensitive', metadata.Collection.tags_case_insensitive ? 'true' : 'false');
    // Build the final URL – the Files browsing route is `/files`
    return `/files?${params.toString()}`;
  };

  return (
    <>
      <OverlayTipBottom tip={`View files in collection "${entity.name}"`}>
        {/*@ts-ignore*/}
        <Button className="icon-btn mx-1" variant="" disabled={false} onClick={() => navigate(buildFilesUrl(entity))}>
          <FaMagnifyingGlass size={20} />
        </Button>
      </OverlayTipBottom>
    </>
  );
};
