import { JSX, lazy } from 'react';

// project imports
import { Entities, ExtendedTypeMap } from '@models/entities';
const FileDetails = lazy(async () => await import('./override_pages/FileDetails'));
const RepoDetails = lazy(async () => await import('./override_pages/RepoDetails'));

export const EntityDetailsRoutes: Record<
  string,
  { type: keyof ExtendedTypeMap; override_page?: React.LazyExoticComponent<() => JSX.Element> }
> = {
  '/collection/:entityID': { type: Entities.Collection },
  '/device/:entityID': { type: Entities.Device },
  '/file': { type: Entities.File, override_page: FileDetails },
  '/file/:sha256': { type: Entities.File, override_page: FileDetails },
  '/filesystem/:entityID': { type: Entities.FileSystem },
  '/folder/:entityID': { type: Entities.Folder },
  '/network/connection/:entityID': { type: Entities.NetworkConnection },
  '/other/:entityID': { type: Entities.Other },
  '/repo/*': { type: Entities.Repo, override_page: RepoDetails },
  '/rules/sigma/:entityID': { type: Entities.SigmaRule },
  '/vendor/:entityID': { type: Entities.Vendor },
  '/windows/process/tree/:entityID': { type: Entities.WindowsProcessTree },
  '/windows/process/:entityID': { type: Entities.WindowsProcess },
};

export const EntityDetailsBasePathByType: Partial<Record<Entities, string>> = Object.entries(EntityDetailsRoutes).reduce(
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

export function getDetailsBasePathByEntity(entity: keyof ExtendedTypeMap): string | undefined {
  return EntityDetailsBasePathByType[entity];
}
