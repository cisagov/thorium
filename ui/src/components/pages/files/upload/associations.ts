// project imports
import { EntityTypes } from '@models/entities';
import type { AssociationCreate, AssociationTarget } from './types';
import { BlankAssociationCreate } from './types';
import { AssociationKind, associationKindLabel, CONTAINER_ASSOCIATION_KINDS } from '@models/associations';
import { createAssociation } from '@thorpi/associations';

// spec: ./upload.spec.md

/**
 * Association kinds surfaced first on the upload page's picker, in priority order.
 *
 * These are the relationships that most commonly describe an uploaded file's link to an entity, so
 * they are grouped ahead of the rest of the (less file-relevant) kinds.
 */
export const FILE_UPLOAD_ASSOCIATION_KINDS: AssociationKind[] = [
  AssociationKind.FileIn,
  AssociationKind.FileFor,
  AssociationKind.FirmwareFor,
  AssociationKind.DocumentationFor,
  AssociationKind.AssociatedWith,
];

/**
 * Association kinds the backend does not yet recognize. Excluded from the upload picker so a user
 * can't pick a kind the API would reject.
 */
const UNSUPPORTED_ASSOCIATION_KINDS: AssociationKind[] = [AssociationKind.SectionIn, AssociationKind.ImportIn];

/**
 * A labeled group of association kinds for the upload picker's `<optgroup>`s.
 */
export interface AssociationKindOptionGroup {
  /** The group heading. */
  label: string;
  /** The association kinds in this group, in display order. */
  kinds: AssociationKind[];
}

/**
 * Build the grouped association-kind options for the upload page picker.
 *
 * The file-relevant kinds ({@link FILE_UPLOAD_ASSOCIATION_KINDS}) come first in priority order, then
 * every remaining supported kind sorted by its human-readable label. Backend-unsupported kinds are
 * omitted entirely.
 *
 * @returns The ordered `[File uploads, Other]` option groups.
 */
export function getGroupedAssociationKinds(): AssociationKindOptionGroup[] {
  const prioritized = new Set<AssociationKind>(FILE_UPLOAD_ASSOCIATION_KINDS);
  const unsupported = new Set<AssociationKind>(UNSUPPORTED_ASSOCIATION_KINDS);
  // everything not already prioritized and not backend-unsupported, alphabetized by display label
  const other = Object.values(AssociationKind)
    .filter((kind) => !prioritized.has(kind) && !unsupported.has(kind))
    .sort((a, b) => associationKindLabel(a).localeCompare(associationKindLabel(b)));
  return [
    { label: 'File uploads', kinds: FILE_UPLOAD_ASSOCIATION_KINDS },
    { label: 'Other', kinds: other },
  ];
}

/**
 * Build the pending upload→entity associations for the chosen kinds.
 *
 * The uploaded file's sha256 is not known until after the upload completes, so each pending association
 * only records its kind, groups, and the entity it links to; the file and the source/target direction are
 * filled in later by {@link createFileAssociations}. The entity is stashed in `source` as a carrier — the
 * final direction is decided per kind at creation time, not here.
 *
 * @param associationKeys - The selected association kind values (raw enum strings, possibly label-spaced).
 * @param entity - The entity the uploaded files link to, or `undefined` when uploading unlinked.
 * @param groups - The groups the associations belong to.
 * @param updatePendingAssociations - Setter that stores the built pending associations.
 */
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
      // stash the entity as a carrier; createFileAssociations resolves source vs. target per kind
      newAssociation.source = { Entity: { id: entity.id, name: entity.name } };
      newAssociation.groups = groups;
      newAssociationList.push(newAssociation);
    });
    updatePendingAssociations(newAssociationList);
  }
}

/**
 * Resolve the source/target ends of an upload association from its kind.
 *
 * Container kinds ({@link CONTAINER_ASSOCIATION_KINDS}, e.g. `FileIn`) are created source→target with the
 * container as the source, so the entity (the container) stays the source and the uploaded file becomes the
 * target. Every other kind reads "file <kind> entity" (e.g. `FirmwareFor`: the file is firmware for the
 * device), so the file must be the source and the entity the target — the reverse of the container layout.
 *
 * @param kind - The association kind being created.
 * @param sha256 - The uploaded file's sha256.
 * @param entity - The entity target carried on the pending association's `source`.
 * @returns The `{ source, targets }` pair oriented correctly for the kind.
 */
function orientAssociation(
  kind: AssociationKind,
  sha256: string,
  entity: AssociationTarget,
): { source: AssociationTarget; targets: AssociationTarget[] } {
  if (CONTAINER_ASSOCIATION_KINDS.has(kind)) {
    return { source: entity, targets: [{ File: sha256 }] };
  }
  return { source: { File: sha256 }, targets: [entity] };
}

/**
 * Create the chosen associations between an uploaded file and its entity.
 *
 * Called after the file upload succeeds (once its sha256 is known). Each pending association is oriented per
 * its kind via {@link orientAssociation} so the relationship reads in the intended direction.
 *
 * @param sha256 - The uploaded file's sha256.
 * @param groups - The groups to create the associations in.
 * @param associations - The pending associations built by {@link handleAssociationUpdate}.
 * @param errorHandler - Called with a formatted message if creating an association fails.
 */
export async function createFileAssociations(
  sha256: string,
  groups: string[],
  associations: AssociationCreate[],
  errorHandler: (error: string) => void,
): Promise<void> {
  for (const association of associations) {
    const { source, targets } = orientAssociation(association.kind, sha256, association.source);
    const copy = { ...association, groups, source, targets };
    await createAssociation(copy, errorHandler);
  }
}
