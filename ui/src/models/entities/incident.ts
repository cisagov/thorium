// project imports
import { CreateEntity, Entities, Entity } from './entities';

/// An incident groups related activity together (e.g. an investigation) by the teams, networks,
/// machines, and locations involved. Mirrors the Rust `Incident` entity (`api/src/models/entities/incident.rs`).
export type IncidentMetaFields = {
  /// The (optional) cover term / codename for this incident
  cover_term: string | null;
  /// The mission teams involved in this incident
  mission_teams: string[];
  /// The networks involved in this incident
  networks: string[];
  /// The machines involved in this incident
  machines: string[];
  /// The physical locations involved in this incident
  locations: string[];
};

/// Create metadata matches the response shape — incidents carry only plain scalar/list fields.
export type IncidentCreateMetaFields = IncidentMetaFields;

export type IncidentMeta = {
  Incident: IncidentMetaFields;
};

export type IncidentCreateMeta = {
  Incident: IncidentCreateMetaFields;
};

export type Incident = Entity<Entities.Incident>;

export type CreateIncident = CreateEntity<Entities.Incident>;

export const BlankIncident: Incident = {
  id: '',
  name: '',
  groups: [],
  description: null,
  kind: Entities.Incident,
  metadata: {
    Incident: {
      cover_term: null,
      mission_teams: [],
      networks: [],
      machines: [],
      locations: [],
    },
  },
  tags: {},
  submitter: '',
  created: '',
  image: null,
};

export const BlankCreateIncident: CreateIncident = {
  name: '',
  groups: [],
  tags: {},
  description: null,
  kind: Entities.Incident,
  metadata: {
    Incident: {
      cover_term: null,
      mission_teams: [],
      networks: [],
      machines: [],
      locations: [],
    },
  },
};
