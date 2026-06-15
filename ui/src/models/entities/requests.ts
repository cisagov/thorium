/// Kind-tagged entity metadata as it appears in a downloaded entities result-file.
///
/// Object variants are keyed by entity kind (e.g. `{ Device: {...} }`); unit variants
/// (`Other`, `WindowsProcessTree`) are serialized as a bare string.
export type EntityRequestMetadata = Record<string, unknown> | string;

/// A request to create an entity, mirroring the Rust `EntityRequest`
/// (`api/src/models/entities.rs`). Produced by tools and returned by the
/// per-kind entities result-file route.
export interface EntityRequest {
  /// The entity's name
  name: string;
  /// The kind-tagged metadata for this entity
  metadata: EntityRequestMetadata;
  /// The groups this entity should be in
  groups: string[];
  /// The tags for this entity (key -> values)
  tags: Record<string, string[]>;
  /// An optional description of this entity
  description: string | null;
}

/**
 * Resolve the entity kind for an {@link EntityRequest} from its metadata.
 *
 * Unit-variant metadata is a bare string (the kind itself); object-variant metadata is keyed by the
 * kind (e.g. `{ Device: {...} }`).
 *
 * @param req - The entity request.
 * @returns The entity kind string (e.g. `Device`, `Other`), or `''` when it can't be determined.
 */
export function entityRequestKind(req: EntityRequest): string {
  if (typeof req.metadata === 'string') return req.metadata;
  const keys = Object.keys(req.metadata);
  return keys.length > 0 ? keys[0] : '';
}
