/// The mode a pipeline form section is rendered in
export enum PipelineFormMode {
  /// Creating a brand new pipeline
  Create = 'create',
  /// Creating a new pipeline seeded from an existing one
  Copy = 'copy',
  /// Editing an existing pipeline
  Edit = 'edit',
  /// Read-only display of an existing pipeline
  View = 'view',
}
