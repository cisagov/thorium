//! Adds support for tracking git repositories within Thorium

mod branches;
mod commits;
mod repos;

pub use commits::{
    BranchRequest, Commit, CommitDetails, CommitListOpts, CommitRequest, Commitish,
    CommitishDetails, CommitishKinds, CommitishListParams, CommitishMapRequest, CommitishRequest,
    GitTag, GitTagDetails, GitTagRequest,
};
pub use repos::{
    Repo, RepoCheckout, RepoCreateResponse, RepoDataUploadResponse, RepoDependency,
    RepoDependencyRequest, RepoDownloadOpts, RepoListLine, RepoListOpts, RepoListParams,
    RepoRequest, RepoScheme, RepoSubmission, RepoSubmissionChunk, RepoUrlComponents, TarredRepo,
    UntarredRepo,
};

pub use branches::{Branch, BranchDetails};

#[cfg(feature = "api")]
pub use repos::RepoDataForm;
