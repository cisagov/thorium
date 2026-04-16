import { matchPath, useLocation } from 'react-router';

// project imports
import { EntityBrowsingPages } from '@components/entities/browsing/configs/config';
import { EntityBrowsingRoutes } from '@components/entities/browsing/EntityBrowsingRoutes';
import { Entities } from '@models/entities/entities';

// derive the entity enum name from the resource location
const getEntityTypeFromPath = (pathname: string) => {
  const entries = Object.entries(EntityBrowsingRoutes).sort(([a], [b]) => b.length - a.length);
  for (const [routePattern, entityType] of entries) {
    if (matchPath({ path: routePattern, end: true }, pathname)) {
      return entityType;
    }
  }
  return Entities.Other;
};

const EntityBrowsing = () => {
  // determine entity type from the url location
  const location = useLocation();
  // determine entity type from the url location
  const type = getEntityTypeFromPath(location.pathname);
  const BrowsingPage = EntityBrowsingPages[type];
  return <BrowsingPage />;
};

export default EntityBrowsing;
