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

export interface SectionItem {
  key: string;
  content: ReactNode;
  estimatedHeight: number;
}

const ROW_HEIGHT = 24;
const SECTION_PADDING = 20;

export function estimateRows(rows: number): number {
  return SECTION_PADDING + rows * ROW_HEIGHT;
}

export function distributeSections(sections: SectionItem[]): { left: SectionItem[]; right: SectionItem[] } {
  const left: SectionItem[] = [];
  const right: SectionItem[] = [];
  let leftHeight = 0;
  let rightHeight = 0;

  for (const item of sections) {
    if (leftHeight <= rightHeight) {
      left.push(item);
      leftHeight += item.estimatedHeight;
    } else {
      right.push(item);
      rightHeight += item.estimatedHeight;
    }
  }

  return { left, right };
}

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
