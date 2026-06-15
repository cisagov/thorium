// project imports
import { buildPathByType } from '@entities/shared/routePaths';
import { Entities, ExtendedTypeMap } from '@models/entities';

// spec: ./EntityBrowsing.spec.md

export const EntityBrowsingRoutes: Record<string, Entities> = {
  '/collections': Entities.Collection,
  '/collections/*': Entities.Collection,
  '/devices': Entities.Device,
  '/devices/*': Entities.Device,
  '/files': Entities.File,
  '/filesystems': Entities.FileSystem,
  '/filesystems/*': Entities.FileSystem,
  '/flags': Entities.Flag,
  '/flags/*': Entities.Flag,
  '/incidents': Entities.Incident,
  '/incidents/*': Entities.Incident,
  '/functions/compiled': Entities.CompiledFunction,
  '/functions/compiled/*': Entities.CompiledFunction,
  '/functions/decompiled': Entities.DecompiledFunction,
  '/functions/decompiled/*': Entities.DecompiledFunction,
  '/pe/sections': Entities.PeSection,
  '/pe/sections/*': Entities.PeSection,
  '/pe/imports': Entities.PeImport,
  '/pe/imports/*': Entities.PeImport,
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

export const EntityBrowsingPathByType = buildPathByType(EntityBrowsingRoutes, (type) => type) as Record<Entities, string>;

export function getBrowsingPathByEntity(entity: keyof ExtendedTypeMap): string {
  return EntityBrowsingPathByType[entity];
}
