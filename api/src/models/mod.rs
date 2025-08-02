//! Wrappers for all objects within Thorium

mod bans;
pub mod conversions;
pub mod cursors;
pub mod deadlines;
pub mod elastic;
mod errors;
pub mod events;
pub mod files;
pub mod git;
pub mod groups;
pub mod helpers;
pub mod images;
pub mod jobs;
pub mod logs;
pub mod network_policies;
pub mod notifications;
pub mod pipelines;
pub mod reactions;
pub mod requisitions;
pub mod results;
mod scylla_utils;
pub mod search;
pub mod streams;
pub mod system;
pub mod tags;
mod trees;
pub mod users;
mod version;
mod volumes;

pub use deadlines::Deadline;
pub use elastic::{ElasticDoc, ElasticIndex, ElasticSearchOpts, ElasticSearchParams};
pub use errors::InvalidEnum;
pub use events::{
    Event, EventCacheStatus, EventCacheStatusOpts, EventData, EventIds, EventMarks, EventPopOpts,
    EventRequest, EventTrigger, EventType, TriggerPotential,
};
pub use files::{
    Attachment, Buffer, CartedSample, CarvedOrigin, CarvedOriginTypes, Comment, CommentRequest,
    CommentResponse, DeleteCommentParams, DeleteSampleParams, DownloadedSample, FileDeleteOpts,
    FileDownloadOpts, FileListOpts, FileListParams, Origin, OriginRequest, OriginTypes,
    PcapNetworkProtocol, Sample, SampleCheck, SampleCheckResponse, SampleListLine, SampleRequest,
    SampleSubmissionResponse, Submission, SubmissionChunk, SubmissionUpdate, Tag, TagMap,
    ZipDownloadParams,
};
pub use git::{
    Branch, BranchDetails, BranchRequest, Commit, CommitDetails, CommitListOpts, CommitRequest,
    Commitish, CommitishDetails, CommitishKinds, CommitishListParams, CommitishMapRequest,
    CommitishRequest, GitTag, GitTagDetails, GitTagRequest, Repo, RepoCheckout, RepoCreateResponse,
    RepoDataUploadResponse, RepoDependency, RepoDependencyRequest, RepoDownloadOpts, RepoListLine,
    RepoListOpts, RepoListParams, RepoRequest, RepoScheme, RepoSubmission, RepoSubmissionChunk,
    RepoUrlComponents, TarredRepo,
};
pub use groups::{
    Group, GroupAllowAction, GroupAllowed, GroupAllowedUpdate, GroupDetailsList, GroupList,
    GroupListParams, GroupMap, GroupRequest, GroupStats, GroupUpdate, GroupUsers,
    GroupUsersRequest, GroupUsersUpdate, Roles,
};
pub use images::{
    ArgStrategy, ChildFilters, ChildFiltersUpdate, ChildrenDependencySettings,
    ChildrenDependencySettingsUpdate, Cleanup, CleanupUpdate, Dependencies, DependenciesUpdate,
    DependencyPassStrategy, DependencySettingsUpdate, EphemeralDependencySettings,
    EphemeralDependencySettingsUpdate, Image, ImageArgs, ImageArgsUpdate, ImageBan, ImageBanKind,
    ImageBanUpdate, ImageDetailsList, ImageJobInfo, ImageLifetime, ImageList, ImageListParams,
    ImageNetworkPolicyUpdate, ImageRequest, ImageScaler, ImageUpdate, ImageVersion, Kvm, KvmUpdate,
    KwargDependency, RepoDependencySettings, Resources, ResourcesRequest, ResourcesUpdate,
    ResultDependencySettings, ResultDependencySettingsUpdate, SampleDependencySettings,
    SecurityContext, SecurityContextUpdate, SpawnLimits, TagDependencySettings,
    TagDependencySettingsUpdate,
};
pub use jobs::{
    Checkpoint, GenericJob, GenericJobArgs, GenericJobArgsUpdate, GenericJobKwargs, GenericJobOpts,
    HandleJobResponse, JobDetailsList, JobHandleStatus, JobList, JobListOpts, JobResetRequestor,
    JobResets, JobStatus, RawJob, RunningJob,
};
pub use logs::{Actions, JobActions, ReactionActions, StatusRequest, StatusUpdate};
pub use network_policies::{
    IpBlock, IpBlockRaw, Ipv4Block, Ipv6Block, NetworkPolicy, NetworkPolicyCustomK8sRule,
    NetworkPolicyCustomLabel, NetworkPolicyListLine, NetworkPolicyListOpts,
    NetworkPolicyListParams, NetworkPolicyPort, NetworkPolicyRequest, NetworkPolicyRule,
    NetworkPolicyRuleRaw, NetworkPolicyUpdate, NetworkProtocol,
};
pub use pipelines::{
    Pipeline, PipelineBan, PipelineBanKind, PipelineBanUpdate, PipelineDetailsList, PipelineList,
    PipelineListParams, PipelineRequest, PipelineStats, PipelineUpdate, StageStats,
};
pub use reactions::{
    BulkReactionResponse, HandleReactionResponse, Reaction, ReactionArgs, ReactionCreation,
    ReactionDetailsList, ReactionExpire, ReactionIdResponse, ReactionList, ReactionListParams,
    ReactionRequest, ReactionStatus, ReactionUpdate, StageLogLine, StageLogs, StageLogsAdd,
};
pub use requisitions::{Requisition, ScopedRequisition, SpawnedUpdate};
pub use results::{
    AutoTag, AutoTagLogic, AutoTagUpdate, FilesHandler, FilesHandlerUpdate, OnDiskFile, Output,
    OutputChunk, OutputCollection, OutputCollectionUpdate, OutputDisplayType, OutputHandler,
    OutputResponse, ResultGetParams,
};
pub use search::events::{
    ResultSearchEvent, SearchEvent, SearchEventPopOpts, SearchEventStatus, SearchEventType,
    TagSearchEvent,
};
pub use streams::{Stream, StreamDepth, StreamObj};
pub use system::{
    ActiveJob, Backup, HostPathWhitelistUpdate, Node, NodeGetParams, NodeHealth, NodeListLine,
    NodeListParams, NodeRegistration, NodeUpdate, Pools, ScalerStats, SpawnMap, StreamerInfoUpdate,
    SystemComponents, SystemInfo, SystemInfoParams, SystemSettings, SystemSettingsResetParams,
    SystemSettingsUpdate, SystemSettingsUpdateParams, SystemStats, Worker, WorkerDelete,
    WorkerDeleteMap, WorkerList, WorkerRegistration, WorkerRegistrationList, WorkerStatus,
    WorkerUpdate,
};
pub use trees::{
    Tree, TreeGrowQuery, TreeNode, TreeNodeData, TreeParams, TreeQuery, TreeRelationships,
    TreeSupport,
};
pub use users::{
    AuthResponse, Key, ScrubbedUser, Theme, UnixInfo, User, UserCreate, UserRole, UserSettings,
    UserSettingsUpdate, UserUpdate,
};
pub use version::{Arch, Component, Os, Version};
pub use volumes::{ConfigMap, HostPath, HostPathTypes, Secret, Volume, VolumeTypes, NFS};

// optional imports
pub mod backends;

// client feature reexports
cfg_if::cfg_if! {
    if #[cfg(feature = "client")] {
        pub use git::UntarredRepo;
        pub use cursors::{Cursor, DateOpts};
        pub use files::UncartedSample;
    }
}

// api feature exports
cfg_if::cfg_if! {
    if #[cfg(feature = "api")] {
        pub use cursors::ApiCursor;
        pub use reactions::{RawGenericJobArgs, RawReactionRequest};
        pub use files::{SampleForm, OriginForm, CommentForm};
        pub use git::RepoDataForm;
        pub use jobs::JobReactionIds;
        pub use backends::results::ResultFileDownloadParams;
        pub(crate) use backends::search::events::SearchEventBackend;
    }
}

// api/client reexports
cfg_if::cfg_if! {
    if #[cfg(any(feature = "api", feature = "client"))] {
        pub use tags::{TagDeleteRequest, TagRequest, TagType};
        pub use notifications::{
            Notification, NotificationLevel, NotificationParams, NotificationRequest, NotificationType,
        };
        pub use results::{OutputRequest, OutputKind, OutputMap};
    }
}

// scylla feature reexports
cfg_if::cfg_if! {
    if #[cfg(feature = "scylla-utils")] {
        mod census;

        pub use scylla_utils::repos::{
            CommitishRow, CommitishListRow, RepoTagRow, FullRepoTagRow, RepoRow,
            RepoListRow, CommitData, BranchData, GitTagData,
        };
        pub use scylla_utils::files::{SubmissionListRow, SubmissionRow, CommentRow};
        pub use scylla_utils::results::{OutputId, OutputIdRow, OutputRow, OutputFormBuilder, OutputForm};
        pub use scylla_utils::system::{WorkerRow, NodeRow, WorkerName};
        pub use scylla_utils::tags::{TagRow, FullTagRow, TagListRow};
        pub use scylla_utils::events::EventRow;
        pub use scylla_utils::s3::S3Objects;
        pub use scylla_utils::network_policies::{NetworkPolicyRow, NetworkPolicyListRow};
        pub use census::{CensusSupport, CensusKeys};
        pub use tags::TagCensusCaseInsensitive;

        #[cfg(feature = "rkyv-support")]
        pub use scylla_utils::s3::ArchivedS3Objects;
    }
}

// scylla keys needed for the client
cfg_if::cfg_if! {
    if #[cfg(any(feature = "scylla-utils", feature = "client"))] {
        pub use scylla_utils::keys::{KeySupport, PipelineKey, ImageKey};
    }
}
