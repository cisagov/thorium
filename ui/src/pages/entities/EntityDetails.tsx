import { matchPath, useLocation } from 'react-router';

// project imports
import { EntityDetailsRoutes } from '@components/entities/details/EntityDetailsRoutes';
import { EntityDetailsPages } from '@components/entities/details/configs/configs';
import { Entities, EntityTypeMap } from '@models/entities/entities';

const getEntityTypeFromPath = (pathname: string): keyof EntityTypeMap | Entities.Other => {
  const entries = Object.entries(EntityDetailsRoutes).sort(([a], [b]) => b.length - a.length);
  for (const [routePattern, entityType] of entries) {
    if (matchPath({ path: routePattern, end: true }, pathname)) {
      return entityType.type as keyof EntityTypeMap;
    }
  }
  return Entities.Other;
};

const EntityDetails = () => {
  const location = useLocation();
  // determine entity type from the url location
  const type = getEntityTypeFromPath(location.pathname);
  const DetailsPage = EntityDetailsPages[type];
  return <DetailsPage />;
};

export default EntityDetails;
