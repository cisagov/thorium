import { describe, it, expect, vi, beforeEach } from 'vitest';

// project imports
import { createFileAssociations, FILE_UPLOAD_ASSOCIATION_KINDS, getGroupedAssociationKinds, handleAssociationUpdate } from './associations';
import { createAssociation } from '@thorpi/associations';
import { AssociationKind, associationKindLabel, BlankAssociationCreate } from '@models/associations';
import type { AssociationCreate } from '@models/associations';
import type { EntityTypes } from '@models/entities/entities';

vi.mock('@thorpi/associations', () => ({
  createAssociation: vi.fn().mockResolvedValue(true),
}));

const mockCreateAssociation = vi.mocked(createAssociation);

describe('handleAssociationUpdate', () => {
  it('creates associations when entity is provided', () => {
    const entity = { id: 'e1', name: 'TestEntity', groups: ['g1'] } as unknown as EntityTypes;
    const groups = ['g1', 'g2'];
    const setter = vi.fn();

    handleAssociationUpdate([AssociationKind.AssociatedWith], entity, groups, setter);

    expect(setter).toHaveBeenCalledTimes(1);
    const associations = setter.mock.calls[0][0] as AssociationCreate[];
    expect(associations).toHaveLength(1);
    expect(associations[0].kind).toBe('AssociatedWith');
    expect(associations[0].source).toEqual({ Entity: { id: 'e1', name: 'TestEntity' } });
    expect(associations[0].groups).toEqual(['g1', 'g2']);
  });

  it('creates multiple associations for multiple keys', () => {
    const entity = { id: 'e1', name: 'Test', groups: [] } as unknown as EntityTypes;
    const setter = vi.fn();

    handleAssociationUpdate([AssociationKind.AssociatedWith, AssociationKind.FileFor], entity, ['g1'], setter);

    const associations = setter.mock.calls[0][0] as AssociationCreate[];
    expect(associations).toHaveLength(2);
    expect(associations[0].kind).toBe('AssociatedWith');
    expect(associations[1].kind).toBe('FileFor');
  });

  it('does nothing when entity is undefined', () => {
    const setter = vi.fn();
    handleAssociationUpdate([AssociationKind.AssociatedWith], undefined, ['g1'], setter);
    expect(setter).not.toHaveBeenCalled();
  });

  it('passes empty groups array through', () => {
    const entity = { id: 'e1', name: 'Test', groups: [] } as unknown as EntityTypes;
    const setter = vi.fn();
    handleAssociationUpdate([AssociationKind.AssociatedWith], entity, [], setter);
    const associations = setter.mock.calls[0][0] as AssociationCreate[];
    expect(associations[0].groups).toEqual([]);
  });
});

describe('getGroupedAssociationKinds', () => {
  it('lists the file-upload kinds first in priority order', () => {
    const [fileGroup] = getGroupedAssociationKinds();
    expect(fileGroup.label).toBe('File uploads');
    expect(fileGroup.kinds).toEqual(FILE_UPLOAD_ASSOCIATION_KINDS);
  });

  it('puts all remaining supported kinds in the Other group, sorted by label', () => {
    const [, otherGroup] = getGroupedAssociationKinds();
    expect(otherGroup.label).toBe('Other');
    // none of the prioritized kinds are repeated in Other
    for (const kind of FILE_UPLOAD_ASSOCIATION_KINDS) {
      expect(otherGroup.kinds).not.toContain(kind);
    }
    // Other is alphabetized by human-readable label
    const labels = otherGroup.kinds.map((kind) => associationKindLabel(kind));
    expect(labels).toEqual([...labels].sort((a, b) => a.localeCompare(b)));
  });

  it('excludes backend-unsupported kinds (SectionIn / ImportIn) from every group', () => {
    const all = getGroupedAssociationKinds().flatMap((group) => group.kinds);
    expect(all).not.toContain(AssociationKind.SectionIn);
    expect(all).not.toContain(AssociationKind.ImportIn);
  });
});

describe('createFileAssociations', () => {
  const sha256 = 'a'.repeat(64);
  const entityTarget = { Entity: { id: 'e1', name: 'TestEntity' } };
  // build a pending association the way handleAssociationUpdate does: entity stashed on `source`
  function pending(kind: AssociationKind): AssociationCreate {
    return { ...structuredClone(BlankAssociationCreate), kind, source: entityTarget };
  }

  beforeEach(() => {
    mockCreateAssociation.mockClear();
  });

  it('orients a directional "…For" kind as file→entity (file is the source)', async () => {
    // FirmwareFor reads "file is firmware for the entity", so the file must be the association source
    await createFileAssociations(sha256, ['g1'], [pending(AssociationKind.FirmwareFor)], vi.fn());
    const created = mockCreateAssociation.mock.calls[0][0];
    expect(created.kind).toBe(AssociationKind.FirmwareFor);
    expect(created.source).toEqual({ File: sha256 });
    expect(created.targets).toEqual([entityTarget]);
  });

  it('orients a container kind as entity→file (container entity stays the source)', async () => {
    // FileIn is created container→contained, so the entity (container) stays the source and the file is the target
    await createFileAssociations(sha256, ['g1'], [pending(AssociationKind.FileIn)], vi.fn());
    const created = mockCreateAssociation.mock.calls[0][0];
    expect(created.kind).toBe(AssociationKind.FileIn);
    expect(created.source).toEqual(entityTarget);
    expect(created.targets).toEqual([{ File: sha256 }]);
  });

  it('threads the provided error handler into createAssociation', async () => {
    const errorHandler = vi.fn();
    await createFileAssociations(sha256, ['g1'], [pending(AssociationKind.AssociatedWith)], errorHandler);
    expect(mockCreateAssociation).toHaveBeenCalledWith(expect.anything(), errorHandler);
  });
});
