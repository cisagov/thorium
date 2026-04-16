import { matchPath, useLocation } from 'react-router';

// project imports
import { EntityCreateRoutes } from '@components/entities/create/EntityCreateRoutes';
import { EntityCreatePages } from '@components/entities/create/configs/config';
import { UISupportedEntityCreateKind } from '@models/entities/entities';

// derive the entity enum name from the resource location
const getEntityTypeFromPath = (pathname: string): UISupportedEntityCreateKind | null => {
  const entries = Object.entries(EntityCreateRoutes).sort(([a], [b]) => b.length - a.length);
  for (const [routePattern, entityType] of entries) {
    if (matchPath({ path: routePattern, end: true }, pathname)) {
      return entityType.type as UISupportedEntityCreateKind;
    }
  }
  return null;
};

const EntityCreate = () => {
  const location = useLocation();
  // determine entity type from the url location
  const type = getEntityTypeFromPath(location.pathname);
  // We don't support other, but need a fallback for getEntityTypeFromPath
  if (type === null) {
    return null;
  }
  const CreatePage = EntityCreatePages[type];
  return <CreatePage />;
};

export default EntityCreate;
