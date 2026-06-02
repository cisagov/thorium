import { describe, it, expect, vi } from 'vitest';

// project imports
import { handleAssociationUpdate } from './associations';
import { AssociationKind } from '@models/associations';
import type { AssociationCreate } from '@models/associations';
import type { EntityTypes } from '@models/entities/entities';

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
