import type { ReactNode } from 'react';

// project imports
import { ImageChecker } from '@utilities/rules/tools/image';
import type { ImageBanKind } from '@models/images';

export const imageChecker = new ImageChecker();

export const FIELDS_KEYS = new Set([
  'name',
  'group',
  'version',
  'description',
  'scaler',
  'image',
  'timeout',
  'lifetime',
  'display_type',
  'spawn_limit',
  'collect_logs',
  'generator',
]);

export type EnvValue = Record<string, string | null>;

/** A keyed display-view section tile flowed through the balanced column layout. */
export interface SectionItem {
  /** Stable identity for the section (also the React key of its card). */
  key: string;
  /** The rendered section content. */
  content: ReactNode;
}

/**
 * Render a human-readable description of an image ban for display.
 *
 * @param kind - The ban kind variant to render (generic message, invalid image URL, or invalid host path).
 * @returns A React node describing the ban, or a generic fallback for unrecognized variants.
 */
export function formatBanKind(kind: ImageBanKind): ReactNode {
  if ('Generic' in kind) return <>Ban: {kind.Generic.msg}</>;
  if ('InvalidImageUrl' in kind)
    return (
      <>
        Invalid image URL: <code>{kind.InvalidImageUrl.url}</code>
      </>
    );
  if ('InvalidHostPath' in kind)
    return (
      <>
        Invalid host path: <code>{kind.InvalidHostPath.host_path}</code> (volume: {kind.InvalidHostPath.volume_name})
      </>
    );
  return <>Unknown ban</>;
}
