import { describe, it, expect } from 'vitest';

// project imports
import { ASSOCIATION_KIND_LABELS, AssociationKind, associationKindLabel } from './associations';

describe('ASSOCIATION_KIND_LABELS / associationKindLabel', () => {
  it('has a non-empty label for every AssociationKind value', () => {
    for (const kind of Object.values(AssociationKind)) {
      expect(ASSOCIATION_KIND_LABELS[kind]).toBeTruthy();
    }
  });

  it('preserves acronyms', () => {
    expect(associationKindLabel(AssociationKind.ContainsCVE)).toBe('Contains CVE');
    expect(associationKindLabel(AssociationKind.ContainsCWE)).toBe('Contains CWE');
  });

  it('spaces out multi-word kinds', () => {
    expect(associationKindLabel(AssociationKind.AssociatedWith)).toBe('Associated With');
    expect(associationKindLabel(AssociationKind.ParentCompanyOf)).toBe('Parent Company Of');
    expect(associationKindLabel(AssociationKind.HasNetworkConnection)).toBe('Has Network Connection');
    expect(associationKindLabel(AssociationKind.SectionIn)).toBe('Section In');
    expect(associationKindLabel(AssociationKind.ImportIn)).toBe('Import In');
  });

  it('falls back to humanize for an unknown/string kind', () => {
    expect(associationKindLabel('SomeFutureKind')).toBe('Some Future Kind');
  });
});
