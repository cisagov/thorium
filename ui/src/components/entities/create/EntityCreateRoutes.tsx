import { JSX, lazy } from 'react';

// project imports
import { Entities, ExtendedTypeMap } from '@models/entities';
const FileUpload = lazy(async () => await import('./override_pages/FileUpload'));

export const EntityCreateRoutes: Record<string, { type: Entities; override_page?: React.LazyExoticComponent<() => JSX.Element> }> = {
  '/create/collection': { type: Entities.Collection },
  '/create/collections': { type: Entities.Collection },
  '/create/device': { type: Entities.Device },
  '/create/devices': { type: Entities.Device },
  '/upload': { type: Entities.File, override_page: FileUpload },
  '/upload/*': { type: Entities.File, override_page: FileUpload },
  '/uploads': { type: Entities.File, override_page: FileUpload },
  '/analyze': { type: Entities.File, override_page: FileUpload },
  '/analyze/*': { type: Entities.File, override_page: FileUpload },
  '/analysis': { type: Entities.File, override_page: FileUpload },
  '/create/file': { type: Entities.File, override_page: FileUpload },
  '/create/files': { type: Entities.File, override_page: FileUpload },
  '/create/rule/sigma': { type: Entities.SigmaRule },
  '/create/rules/sigma': { type: Entities.SigmaRule },
  '/create/vendor': { type: Entities.Vendor },
  '/create/vendors': { type: Entities.Vendor },
};

export const EntityCreatePathByType: Partial<Record<Entities, string>> = Object.entries(EntityCreateRoutes).reduce(
  (acc, [path, config]) => {
    const normalizedPath = path.replace(/\/:[^/]+$/, '').replace(/\/\*$/, '');
    const existing = acc[config.type];
    // Prefer a non-wildcard route over a wildcard one
    if (!existing || existing.endsWith('*')) {
      acc[config.type] = normalizedPath;
    }
    return acc;
  },
  {} as Partial<Record<Entities, string>>,
);

export function getCreatePathByEntity(entity: keyof ExtendedTypeMap): string | undefined {
  console.log(entity);
  console.log(EntityCreatePathByType[entity]);
  return EntityCreatePathByType[entity];
}
