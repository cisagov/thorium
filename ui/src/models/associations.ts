// project imports
import { labelWithFallback } from './labels';
import { Direction } from './trees';

/// The different possible associations
export enum AssociationKind {
  /// This file is associated with something else
  FileFor = 'FileFor',
  /// This is documentation for something else
  DocumentationFor = 'DocumentationFor',
  /// This file or repo is or contains firmware for a device
  FirmwareFor = 'FirmwareFor',
  /// This file/repo/entity is associated with something else
  AssociatedWith = 'AssociatedWith',
  /// This was developed or created by
  DevelopedBy = 'DevelopedBy',
  /// This contains a CVE
  ContainsCVE = 'ContainsCVE',
  /// This contains a CWE
  ContainsCWE = 'ContainsCWE',
  /// This is based in specific countries
  BasedIn = 'BasedIn',
  /// This person was or is employed by
  EmployedBy = 'EmployedBy',
  /// This is the parent company of another company
  ParentCompanyOf = 'ParentCompanyOf',
  /// This is used by a specific person or group
  UsedBy = 'UsedBy',
  /// This was used in a specific campaign or engagement
  UsedIn = 'UsedIn',
  /// This campaign was performed by
  PerformedBy = 'PerformedBy',
  /// This filesystem was extracted/carved from
  FileSystemIn = 'FileSystemIn',
  /// This is a folder within a filesystem or another folder
  FolderIn = 'FolderIn',
  /// This is a file in a folder in a filesytem
  FileIn = 'FileIn',
  /// A Process tree in or from something
  ProcessTreeIn = 'ProcessTreeIn',
  /// A Process in a process tree or a child process
  ChildProcess = 'ChildProcess',
  /// Opens or receives data from a network connection
  HasNetworkConnection = 'HasNetworkConnection',
  /// A Sigma rule hit
  SigmaRuleHit = 'SigmaRuleHit',
  /// A PE section within a file/binary
  SectionIn = 'SectionIn',
  /// A library imported by a file/binary
  ImportIn = 'ImportIn',
  /// A flag flags a specific entity (e.g. a Flag flags the WindowsProcess it was raised on)
  FlagFor = 'FlagFor',
  /// This was created by something else (e.g. a Flag created by the SigmaRule that produced it)
  CreatedBy = 'CreatedBy',
}

/**
 * Human-readable display labels for each association kind, colocated with the {@link AssociationKind}
 * enum. Typed as an exhaustive `Record<AssociationKind, string>`: adding a new kind without a label is
 * a compile-time error. Acronyms (CVE/CWE) are preserved. Use for DISPLAY only — the raw enum value is
 * still the API/select value.
 */
export const ASSOCIATION_KIND_LABELS: Record<AssociationKind, string> = {
  [AssociationKind.FileFor]: 'File For',
  [AssociationKind.DocumentationFor]: 'Documentation For',
  [AssociationKind.FirmwareFor]: 'Firmware For',
  [AssociationKind.AssociatedWith]: 'Associated With',
  [AssociationKind.DevelopedBy]: 'Developed By',
  [AssociationKind.ContainsCVE]: 'Contains CVE',
  [AssociationKind.ContainsCWE]: 'Contains CWE',
  [AssociationKind.BasedIn]: 'Based In',
  [AssociationKind.EmployedBy]: 'Employed By',
  [AssociationKind.ParentCompanyOf]: 'Parent Company Of',
  [AssociationKind.UsedBy]: 'Used By',
  [AssociationKind.UsedIn]: 'Used In',
  [AssociationKind.PerformedBy]: 'Performed By',
  [AssociationKind.FileSystemIn]: 'File System In',
  [AssociationKind.FolderIn]: 'Folder In',
  [AssociationKind.FileIn]: 'File In',
  [AssociationKind.ProcessTreeIn]: 'Process Tree In',
  [AssociationKind.ChildProcess]: 'Child Process',
  [AssociationKind.HasNetworkConnection]: 'Has Network Connection',
  [AssociationKind.SigmaRuleHit]: 'Sigma Rule Hit',
  [AssociationKind.SectionIn]: 'Section In',
  [AssociationKind.ImportIn]: 'Import In',
  [AssociationKind.FlagFor]: 'Flag For',
  [AssociationKind.CreatedBy]: 'Created By',
};

/**
 * Get the human-readable label for an association kind.
 *
 * @param kind - An {@link AssociationKind} value, or a raw kind string (some call sites carry `string`).
 * @returns The mapped label, falling back to {@link humanize} for an unknown/unmapped kind.
 */
export function associationKindLabel(kind: AssociationKind | string): string {
  return labelWithFallback(ASSOCIATION_KIND_LABELS, kind);
}

/**
 * Containment ("… In …") association kinds whose relationship badge should name the container it links to
 * (e.g. "File In somefolder Folder"). These are created source→target with the **source as the container**
 * (see the filesystem entity builders in the API), so the container is the association's counterpart. Kept
 * as an explicit set — NOT an `endsWith('In')` heuristic, which would wrongly match `BasedIn`/`UsedIn`.
 */
export const CONTAINER_ASSOCIATION_KINDS: ReadonlySet<AssociationKind> = new Set([
  AssociationKind.FileSystemIn,
  AssociationKind.FolderIn,
  AssociationKind.FileIn,
  AssociationKind.ProcessTreeIn,
  AssociationKind.SectionIn,
  AssociationKind.ImportIn,
]);

/**
 * Non-structural ("relationship") association kinds — the whitelist of kinds eligible to be surfaced
 * *against* their stored direction (a node showing what points **at** it, e.g. a `Flag`/`SigmaRule` under the
 * `WindowsProcess` they hit). **Every other association kind is structural (directional-only) by default**, so
 * a newly added kind is safe: it forms hierarchy (descends/hoists) and is never fanned out in reverse until it
 * is explicitly added here. This single set drives both the tree views' reverse-display eligibility
 * ({@link defaultBidirectional}) and the overlay's hoisting spine (structural = not in this set).
 */
export const NON_STRUCTURAL_ASSOCIATION_KINDS: ReadonlySet<AssociationKind> = new Set<AssociationKind>([
  AssociationKind.SigmaRuleHit,
  AssociationKind.HasNetworkConnection,
  AssociationKind.ContainsCVE,
  AssociationKind.ContainsCWE,
  AssociationKind.AssociatedWith,
  // the sigma→flag→process chain (`SigmaRule -CreatedBy-> Flag -FlagFor-> WindowsProcess`): these are
  // relationship edges so a flagged process reveals its Flags, and each Flag reveals the SigmaRule that raised it
  AssociationKind.FlagFor,
  AssociationKind.CreatedBy,
]);

export type AssociationTarget = {
  /// This association is associated with another entity
  Entity?: { id: string; name: string };
  /// This association is associated with a file
  File?: string;
  /// This association is associated with a repo
  Repo?: string;
};

export type Association = {
  /// The kind of association this is
  kind: AssociationKind;
  /// The other data this directional association is with
  other: AssociationTarget;
  /// The creator of this association
  submitter: string;
  /// The groups for this association
  groups: string[];
  /// When this association was created
  created: string;
  /// The direction for this association
  direction: Direction;
};

export type AssociationCreate = {
  /// The kind of association to make
  kind: AssociationKind;
  /// The piece of data this association starts with
  source: AssociationTarget;
  /// The data this association is with
  targets: AssociationTarget[];
  /// The groups for this association
  groups: string[];
  /// Whether this is a bidirectional relationship or not
  is_bidirectional: boolean;
};

export const BlankAssociationCreate: AssociationCreate = {
  kind: AssociationKind.AssociatedWith,
  source: {},
  targets: [],
  groups: [],
  is_bidirectional: false,
};
