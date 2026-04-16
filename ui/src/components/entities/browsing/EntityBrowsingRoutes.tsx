// project imports
import { Entities, ExtendedTypeMap } from '@models/entities';

export const EntityBrowsingRoutes: Record<string, Entities> = {
  '/collections': Entities.Collection,
  '/collections/*': Entities.Collection,
  '/devices': Entities.Device,
  '/devices/*': Entities.Device,
  '/files': Entities.File,
  '/filesystems': Entities.FileSystem,
  '/filesystems/*': Entities.FileSystem,
  '/folders': Entities.Folder,
  '/folders/*': Entities.Folder,
  '/network/connections': Entities.NetworkConnection,
  '/network/connections/*': Entities.NetworkConnection,
  '/repos': Entities.Repo,
  '/repos/*': Entities.Repo,
  '/rules/sigma': Entities.SigmaRule,
  '/rules/sigma/*': Entities.SigmaRule,
  '/vendors': Entities.Vendor,
  '/vendors/*': Entities.Vendor,
  '/windows/process/trees': Entities.WindowsProcessTree,
  '/windows/process/trees/*': Entities.WindowsProcessTree,
  '/windows/processes': Entities.WindowsProcess,
  '/windows/processes/*': Entities.WindowsProcess,
};

export const EntityBrowsingPathByType: Record<Entities, string> = Object.entries(EntityBrowsingRoutes).reduce(
  (acc, [path, type]) => {
    const normalizedPath = path.replace(/\/:[^/]+$/, '').replace(/\/\*$/, '');
    const existing = acc[type];
    // Prefer a non-wildcard route over a wildcard one
    if (!existing || existing.endsWith('*')) {
      acc[type] = normalizedPath;
    }
    return acc;
  },
  {} as Record<Entities, string>,
);

export function getBrowsingPathByEntity(entity: keyof ExtendedTypeMap): string {
  return EntityBrowsingPathByType[entity];
}
