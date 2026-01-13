//! The arguments for backups

use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// The backup specific subcommands
#[derive(Parser, Debug, Clone)]
pub enum BackupSubCommands {
    /// Take a new backup
    #[clap(version, author)]
    New(NewBackup),
    /// Scrub a backup for bitrot
    Scrub(ScrubBackup),
    /// Restore a backup to a Thorium cluster
    #[clap(version, author)]
    Restore(RestoreBackup),
}

/// Define the default backup components
fn default_backup_components() -> Vec<BackupComponents> {
    // return all by default
    vec![BackupComponents::All]
}

/// The tables that can be backed up
#[derive(Debug, Clone, Copy, strum::Display, strum::EnumIter, ValueEnum, PartialEq)]
#[strum(serialize_all = "kebab-case")]
pub enum BackupComponents {
    /// Backup all components
    All,
    /// Backup all components except those in S3
    NoS3,
    Redis,
    Associations,
    CommentAttachments,
    Comments,
    Commitish,
    CommitishList,
    Entities,
    NetworkPolicies,
    Nodes,
    Notifications,
    RepoData,
    RepoList,
    ResultFiles,
    Results,
    ResultsStream,
    S3Ids,
    S3IdsObjects,
    SamplesList,
    Tags,
}

impl BackupComponents {
    /// Returns true if the component is just a filter rather than an actual component
    pub fn is_filter(self) -> bool {
        matches!(self, BackupComponents::All | BackupComponents::NoS3)
    }

    /// Returns true if the component is stored in S3
    fn is_in_s3(self) -> bool {
        matches!(
            self,
            BackupComponents::CommentAttachments
                | BackupComponents::ResultFiles
                | BackupComponents::S3IdsObjects
        )
    }

    /// Determine whether the component should be backed up
    ///
    /// # Arguments
    ///
    /// * `components` - The list of components to back up supplied by the user
    pub fn should_backup(self, components: &[BackupComponents]) -> bool {
        // check if we should backup all or backup just in S3;
        // we'll check inside this function for clarity; O(n) lookups over a
        // tiny array like this shouldn't have any meaningful performance impact
        let backup_all = components.contains(&BackupComponents::All);
        let no_s3 = components.contains(&BackupComponents::NoS3);
        // determine whether we should backup this component
        match (
            backup_all,
            components.contains(&self),
            no_s3,
            self.is_in_s3(),
        ) {
            // always backup if we're set to backup all
            (true, _, _, _) |
            // backup if the component was explicitly provided
            (_, true, _, _) |
            // not explicitly provided, but backup if no_s3 and it's *not* in S3
            (_, _, true, false) => true,
            // otherwise don't backup
            _ => false,
        }
    }
}

/// Backup a Thorium cluster
#[derive(Parser, Debug, Clone)]
pub struct NewBackup {
    /// The components to backup
    ///
    /// 'all' will backup all components while 'no-s3' will backup all components
    /// except for those stored in S3. If both are given, 'all' overrides 'no-s3'.
    #[clap(value_enum, default_values_t = default_backup_components(), value_delimiter = ',')]
    pub components: Vec<BackupComponents>,
    /// Where to store our backups
    #[clap(short, long, default_value = "ThoriumBackups")]
    pub output: PathBuf,
    /// The chunk multiplier to use with our worker count
    #[clap(short, long, default_value = "100")]
    pub multiplier: u64,
}

/// Scrub a backup for bitrot
#[derive(Parser, Debug, Clone)]
pub struct ScrubBackup {
    /// The path to the backup to scrub
    #[clap(short, long)]
    pub backup: PathBuf,
}

/// Restore a backup to a specific Thorium cluster
#[derive(Parser, Debug, Clone)]
pub struct RestoreBackup {
    /// The path to the backup to restore
    #[clap(short, long)]
    pub backup: PathBuf,
}

#[cfg(test)]
mod tests {
    use strum::IntoEnumIterator;

    use super::BackupComponents;

    #[test]
    fn backup_all_always_true() {
        // when the user asks for `All`, every component should be backed up
        let all = [BackupComponents::All];
        for comp in BackupComponents::iter() {
            assert!(
                comp.should_backup(&all),
                "component {comp} should be backed up when `All` is requested",
            );
        }
        // even if `NoS3` is set
        let all = [BackupComponents::All, BackupComponents::NoS3];
        for comp in BackupComponents::iter() {
            assert!(
                comp.should_backup(&all),
                "component {comp} should be backed up when `All` is requested even if `NoS3` is also set",
            );
        }
    }

    #[test]
    fn no_s3_filters_out_s3_components() {
        // `NoS3` should skip anything that lives in S3 unless explicitly provided
        let set = [BackupComponents::NoS3];
        // check S3 components
        for comp in BackupComponents::iter().filter(|comp| comp.is_in_s3()) {
            assert!(
                !comp.should_backup(&set),
                "S3 component {comp} should be skipped with `NoS3`"
            );
        }
        // check non-S3 components
        for comp in BackupComponents::iter().filter(|comp| !comp.is_in_s3()) {
            assert!(
                comp.should_backup(&set),
                "Non-S3 component {comp} should not be skipped with `NoS3`",
            );
        }
    }

    #[test]
    fn explicit_component() {
        // no `All` and no `NoS3` means only the components the user listed are backed up
        let set = [BackupComponents::CommentAttachments];
        // listed component is true
        assert!(BackupComponents::CommentAttachments.should_backup(&set));
        // anything else is false
        for comp in
            BackupComponents::iter().filter(|comp| *comp != BackupComponents::CommentAttachments)
        {
            assert!(
                !comp.should_backup(&set),
                "component {comp} should NOT be backed up when not explicitly requested",
            );
        }
    }

    #[test]
    fn explicit_non_s3_with_no_s3_flag() {
        // `NoS3` + an explicit non‑S3 component should be true
        let set = [
            BackupComponents::NoS3,
            // pick a variant that is *not* stored in S3; for demonstration we use `SampleList`
            BackupComponents::SamplesList,
        ];
        assert!(BackupComponents::SamplesList.should_backup(&set));
    }

    #[test]
    fn explicit_s3_with_no_s3_flag() {
        // `NoS3` + an explicit S3 component is true (the explicit request should override the filter)
        let set = [BackupComponents::NoS3, BackupComponents::ResultFiles];
        assert!(BackupComponents::ResultFiles.should_backup(&set));
    }
}
