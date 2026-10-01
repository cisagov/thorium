//! Arguments for association-related Thorctl commands

#![allow(clippy::module_name_repetitions)]

use clap::Parser;
use std::path::PathBuf;
use thorium::Error;
use thorium::models::{AssociationKind, Directionality};
use uuid::Uuid;

/// The commands to send to the associations task handler
#[derive(Parser, Debug)]
pub enum Associations {
    /// Create associations between a file, repo, or entity and other files, repos, or entities
    #[clap(version, author)]
    Create(CreateAssociations),
    /// Delete associations between a file, repo, or entity and other files, repos, or entities
    ///
    /// Only the submitter of an association, a manager/owner of the group it is in,
    /// or an admin can delete an association.
    #[clap(version, author)]
    Delete(DeleteAssociations),
    /// List the associations for a file, repo, or entity
    #[clap(version, author, alias = "get")]
    List(ListAssociations),
}

/// Parse and validate a sha256
///
/// # Arguments
///
/// * `raw` - The raw sha256 to parse
fn parse_sha256(raw: &str) -> Result<String, Error> {
    // make sure this is 64 hex characters
    if raw.len() == 64 && raw.chars().all(|c| c.is_ascii_hexdigit()) {
        Ok(raw.to_ascii_lowercase())
    } else {
        Err(Error::new(format!(
            "'{raw}' is not a valid sha256 (expected 64 hex characters)"
        )))
    }
}

/// Normalize a repo url the same way Thorium does
///
/// This strips any scheme, trailing '/', and trailing '.git'. Nested repos are allowed.
///
/// # Arguments
///
/// * `raw` - The raw repo url to normalize
pub(crate) fn normalize_repo_url(raw: &str) -> Result<String, Error> {
    // strip any scheme
    let url = raw.trim_start_matches("https://");
    let url = url.trim_start_matches("http://");
    // remove any trailing '/' or '.git'
    let url = url.trim_end_matches('/').trim_end_matches(".git");
    // make sure we have a host, user, and name with no empty components
    let mut split = url.split('/');
    if split.clone().count() < 3 || split.any(str::is_empty) {
        return Err(Error::new(format!(
            "'{raw}' is not a valid repo url (expected <HOST>/<USER>/<NAME>)"
        )));
    }
    Ok(url.to_owned())
}

/// The object that associations start from
#[derive(clap::Args, Debug, Clone)]
#[group(required = true, multiple = false)]
pub struct SourceArgs {
    /// The sha256 of the file the associations start from
    #[clap(long, value_parser = parse_sha256)]
    pub file: Option<String>,
    /// The url of the repo the associations start from
    #[clap(long, value_parser = normalize_repo_url)]
    pub repo: Option<String>,
    /// The id of the entity the associations start from
    #[clap(long)]
    pub entity: Option<Uuid>,
}

/// A file, repo, or entity in Thorium
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssocObject {
    /// A file by sha256
    File(String),
    /// A repo by url
    Repo(String),
    /// An entity by id
    Entity(Uuid),
}

impl std::fmt::Display for AssocObject {
    /// Cleanly print an association object
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AssocObject::File(sha256) => write!(f, "file {sha256}"),
            AssocObject::Repo(url) => write!(f, "repo {url}"),
            AssocObject::Entity(id) => write!(f, "entity {id}"),
        }
    }
}

impl SourceArgs {
    /// Get the object our associations start from
    ///
    /// # Panics
    ///
    /// Panics if no source was set, which clap prevents
    #[must_use]
    pub fn object(&self) -> AssocObject {
        match (&self.file, &self.repo, &self.entity) {
            (Some(sha256), _, _) => AssocObject::File(sha256.clone()),
            (_, Some(url), _) => AssocObject::Repo(url.clone()),
            (_, _, Some(id)) => AssocObject::Entity(*id),
            (None, None, None) => unreachable!("clap requires a source to be set"),
        }
    }
}

/// The objects that associations point to
#[derive(clap::Args, Debug, Clone)]
#[group(required = true, multiple = true)]
pub struct TargetArgs {
    /// The sha256s of the target files
    #[clap(short = 'F', long, value_delimiter = ',', value_parser = parse_sha256)]
    pub target_files: Vec<String>,
    /// The urls of the target repos
    #[clap(short = 'R', long, value_delimiter = ',', value_parser = normalize_repo_url)]
    pub target_repos: Vec<String>,
    /// The ids of the target entities
    #[clap(short = 'E', long, value_delimiter = ',')]
    pub target_entities: Vec<Uuid>,
}

impl TargetArgs {
    /// Get all of the objects our associations point to
    #[must_use]
    pub fn objects(&self) -> Vec<AssocObject> {
        self.target_files
            .iter()
            .map(|sha256| AssocObject::File(sha256.clone()))
            .chain(
                self.target_repos
                    .iter()
                    .map(|url| AssocObject::Repo(url.clone())),
            )
            .chain(
                self.target_entities
                    .iter()
                    .map(|id| AssocObject::Entity(*id)),
            )
            .collect()
    }
}

/// Create associations between some data in Thorium
#[derive(Parser, Debug, Clone)]
pub struct CreateAssociations {
    /// The object these associations start from
    #[command(flatten)]
    pub source: SourceArgs,
    /// The objects these associations point to
    #[command(flatten)]
    pub targets: TargetArgs,
    /// The kind of association to create
    #[clap(short, long, value_enum, ignore_case = true)]
    pub kind: AssociationKind,
    /// Make these associations bidirectional
    #[clap(short, long)]
    pub bidirectional: bool,
    /// The groups to create these associations in
    ///
    /// Defaults to the groups shared by the source and each target.
    #[clap(short, long, value_delimiter = ',')]
    pub groups: Vec<String>,
}

/// The direction of an association relative to its source
#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectionArg {
    /// The association points from the source to the target
    To,
    /// The association points from the target to the source
    From,
    /// The association points both ways
    Bidirectional,
}

impl From<DirectionArg> for Directionality {
    /// Convert a direction arg into a [`Directionality`]
    fn from(arg: DirectionArg) -> Self {
        match arg {
            DirectionArg::To => Directionality::To,
            DirectionArg::From => Directionality::From,
            DirectionArg::Bidirectional => Directionality::Bidirectional,
        }
    }
}

/// Delete associations between some data in Thorium
#[derive(Parser, Debug, Clone)]
pub struct DeleteAssociations {
    /// The object the associations to delete start from
    ///
    /// Associations are matched regardless of which side created them.
    #[command(flatten)]
    pub source: SourceArgs,
    /// The objects to delete associations with
    #[command(flatten)]
    pub targets: TargetArgs,
    /// The kind of association to delete
    #[clap(short, long, value_enum, ignore_case = true)]
    pub kind: AssociationKind,
    /// The groups to delete these associations from (defaults to all of your groups)
    #[clap(short, long, value_delimiter = ',')]
    pub groups: Vec<String>,
    /// Only delete associations in these directions relative to the source (defaults to any)
    #[clap(short, long, value_enum, ignore_case = true, value_delimiter = ',')]
    pub direction: Vec<DirectionArg>,
    /// Skip the confirmation prompt before deleting
    #[clap(short = 'y', long)]
    pub skip_confirm: bool,
}

/// List the associations for some data in Thorium
#[derive(Parser, Debug, Clone)]
pub struct ListAssociations {
    /// The object to list associations for
    #[command(flatten)]
    pub source: SourceArgs,
    /// Only show associations of these kinds
    ///
    /// This is filtered after listing so it is applied after `--limit`.
    #[clap(short, long, value_enum, ignore_case = true, value_delimiter = ',')]
    pub kind: Vec<AssociationKind>,
    /// The groups to list associations from (defaults to all of your groups)
    #[clap(short, long, value_delimiter = ',')]
    pub groups: Vec<String>,
    /// The max number of associations to list
    #[clap(short, long, default_value = "50", conflicts_with = "no_limit")]
    pub limit: usize,
    /// List all associations
    #[clap(long)]
    pub no_limit: bool,
    /// The number of associations to get per request
    #[clap(short, long, default_value = "50")]
    pub page_size: usize,
    /// Print the associations as JSON instead of a table
    #[clap(long)]
    pub json: bool,
    /// Print condensed JSON (implies `--json`)
    #[clap(short, long)]
    pub condensed: bool,
    /// Write JSON to a file instead of stdout (implies `--json`)
    #[clap(short, long)]
    pub output: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{Args, SubCommands};
    use clap::CommandFactory;

    const SHA256: &str = "63B9B5A8C4D9F2B7CB5D2A1B5E8E1FD4F6D9E9A5C2B1A0F8E7D6C5B4A3928170";
    const UUID: &str = "bf5b4ea2-7a43-4a69-8c41-1f3b2c6e5d8a";

    /// Parse an associations command
    fn parse(args: &[&str]) -> Result<Associations, clap::Error> {
        // add our binary and subcommand names
        let full = ["thorctl", "associations"].iter().chain(args.iter());
        match Args::try_parse_from(full)?.cmd {
            SubCommands::Associations(cmd) => Ok(cmd),
            _ => unreachable!("we always parse an associations command"),
        }
    }

    #[test]
    fn debug_assert() {
        Associations::command().debug_assert();
    }

    #[test]
    fn create() {
        let cmd = parse(&[
            "create",
            "--file",
            SHA256,
            "--kind",
            "fileFOR",
            "-E",
            UUID,
            "-R",
            "https://github.com/curl/curl.git/",
            "-b",
            "-g",
            "a,b",
        ])
        .unwrap();
        let Associations::Create(cmd) = cmd else {
            panic!("expected create command")
        };
        assert_eq!(cmd.kind, AssociationKind::FileFor);
        assert_eq!(
            cmd.source.object(),
            AssocObject::File(SHA256.to_ascii_lowercase())
        );
        assert_eq!(
            cmd.targets.objects(),
            vec![
                AssocObject::Repo("github.com/curl/curl".to_owned()),
                AssocObject::Entity(Uuid::parse_str(UUID).unwrap()),
            ]
        );
        assert!(cmd.bidirectional);
        assert_eq!(cmd.groups, vec!["a", "b"]);
    }

    #[test]
    fn kind_verbatim() {
        let cmd = parse(&[
            "create",
            "--entity",
            UUID,
            "-k",
            "ContainsCVE",
            "-F",
            SHA256,
        ])
        .unwrap();
        let Associations::Create(cmd) = cmd else {
            panic!("expected create command")
        };
        assert_eq!(cmd.kind, AssociationKind::ContainsCVE);
    }

    #[test]
    fn delete() {
        let cmd = parse(&[
            "delete",
            "--repo",
            "github.com/a/b/c",
            "-k",
            "AssociatedWith",
            "-F",
            &format!("{SHA256},{SHA256}"),
            "-d",
            "to,bidirectional",
            "-y",
        ])
        .unwrap();
        let Associations::Delete(cmd) = cmd else {
            panic!("expected delete command")
        };
        assert_eq!(
            cmd.source.object(),
            AssocObject::Repo("github.com/a/b/c".to_owned())
        );
        assert_eq!(cmd.targets.objects().len(), 2);
        assert_eq!(
            cmd.direction
                .into_iter()
                .map(Directionality::from)
                .collect::<Vec<_>>(),
            vec![Directionality::To, Directionality::Bidirectional]
        );
        assert!(cmd.skip_confirm);
    }

    #[test]
    fn list_no_targets() {
        let cmd = parse(&[
            "list",
            "--entity",
            UUID,
            "--no-limit",
            "-k",
            "FlagFor,FileFor",
        ])
        .unwrap();
        let Associations::List(cmd) = cmd else {
            panic!("expected list command")
        };
        assert!(cmd.no_limit);
        assert_eq!(
            cmd.kind,
            vec![AssociationKind::FlagFor, AssociationKind::FileFor]
        );
    }

    #[test]
    fn rejects_invalid() {
        // two sources
        assert!(
            parse(&[
                "create", "--file", SHA256, "--entity", UUID, "-k", "FileFor", "-F", SHA256
            ])
            .is_err()
        );
        // no source
        assert!(parse(&["create", "-k", "FileFor", "-F", SHA256]).is_err());
        // no targets
        assert!(parse(&["create", "--file", SHA256, "-k", "FileFor"]).is_err());
        assert!(parse(&["delete", "--file", SHA256, "-k", "FileFor"]).is_err());
        // no kind
        assert!(parse(&["create", "--file", SHA256, "-F", SHA256]).is_err());
        // bad sha256
        assert!(parse(&["create", "--file", "abc", "-k", "FileFor", "-F", SHA256]).is_err());
        // bad uuid
        assert!(parse(&["create", "--entity", "nope", "-k", "FileFor", "-F", SHA256]).is_err());
        // bad kind
        assert!(parse(&["create", "--file", SHA256, "-k", "NotAKind", "-F", SHA256]).is_err());
        // limit and no limit
        assert!(parse(&["list", "--file", SHA256, "-l", "5", "--no-limit"]).is_err());
    }

    #[test]
    fn repo_urls() {
        assert_eq!(
            normalize_repo_url("https://github.com/x/y.git/").unwrap(),
            "github.com/x/y"
        );
        assert_eq!(
            normalize_repo_url("http://gitlab.com/a/b/c").unwrap(),
            "gitlab.com/a/b/c"
        );
        assert!(normalize_repo_url("github.com/x").is_err());
        assert!(normalize_repo_url("github.com//y").is_err());
    }
}
