// project imports
import { EntityTypes } from '@models/entities';
import type { AssociationCreate } from './types';
import { AssociationKind, BlankAssociationCreate } from './types';
import { createAssociation } from '@thorpi/associations';

export function handleAssociationUpdate(
  associationKeys: string[],
  entity: EntityTypes | undefined,
  groups: string[],
  updatePendingAssociations: (associations: AssociationCreate[]) => void,
): void {
  const newAssociationList: AssociationCreate[] = [];
  if (entity) {
    associationKeys.forEach((type) => {
      const newAssociation = structuredClone(BlankAssociationCreate);
      newAssociation.kind = type.replaceAll(' ', '') as unknown as AssociationKind;
      newAssociation.source = { Entity: { id: entity.id, name: entity.name } };
      newAssociation.groups = groups;
      newAssociationList.push(newAssociation);
    });
    updatePendingAssociations(newAssociationList);
  }
}

export async function createFileAssociations(sha256: string, groups: string[], associations: AssociationCreate[]): Promise<void> {
  for (const association of associations) {
    const copy = { ...association, groups, targets: [{ File: sha256 }] };
    await createAssociation(copy, console.log);
  }
}
