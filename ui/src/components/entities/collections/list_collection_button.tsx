import { Button } from 'react-bootstrap';
import { FaMagnifyingGlass } from 'react-icons/fa6';
import { useNavigate } from 'react-router-dom';

import { OverlayTipBottom } from '@components';
import { Collection, CollectionMeta } from 'models';

export const buildCollectionsBrowsingUrl = (collection: Collection): string => {
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
  return `/${metadata.Collection.collection_kind?.toLocaleLowerCase()}?${params.toString()}`;
};

type ListCollectionProps = {
  collection: Collection;
};

export const ListCollectionButton: React.FC<ListCollectionProps> = ({ collection }) => {
  const navigate = useNavigate();
  console.log(collection);
  return (
    <>
      <OverlayTipBottom tip={`View files in collection "${collection.name}"`}>
        <Button className="icon-btn mx-1" variant="" disabled={false} onClick={() => navigate(buildCollectionsBrowsingUrl(collection))}>
          <FaMagnifyingGlass size={20} />
        </Button>
      </OverlayTipBottom>
    </>
  );
};
