use cidr::{Ipv4Cidr, Ipv6Cidr};
use futures::{StreamExt, TryStreamExt, stream};
use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng, seq::IteratorRandom};
use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::LazyLock;
use uuid::Uuid;

use crate::client::{ClientSettings, Users};
use crate::models::entities::network_activity::{
    NetConState, NetworkConnection, TransportLayerProtocol,
};
use crate::models::entities::rules::{SigmaActionToTake, SigmaAutoFlag};
use crate::models::{
    ArgStrategy, Buffer, BulkReactionResponse, ChildFilters, Cleanup, CollectionEntityRequest,
    CollectionKind, CompiledFunction, CompiledInstruction, Confidence, CriticalSector,
    DecompiledFunction, Dependencies, DependencyPassStrategy, DeviceEntityRequest, Entity,
    EntityMetadata, EntityMetadataRequest, EntityMetadataUpdate, EntityRequest,
    EphemeralDependencySettings, FileSystemEntity,
    FileSystemFolderEntity, FilesHandler, Flag, GenericJobArgs, GroupRequest, GroupUsersRequest,
    ImageLifetime, ImageRequest, ImageScaler, ImageVersion, IncidentRequest, IpBlock, IpBlockRaw,
    Ipv4Block,
    Ipv6Block, KwargDependency, NetworkPolicyCustomK8sRule, NetworkPolicyCustomLabel,
    NetworkPolicyPort, NetworkPolicyRequest, NetworkPolicyRuleRaw, NetworkProtocol,
    NodeRegistration, OriginRequest, OutputCollection, OutputDisplayType, PeImportEntity,
    PeSectionEntity, Pipeline, PipelineRequest, Pools, ReactionCreation, ReactionRequest,
    RepoCheckout, RepoDependencySettings, RepoRequest, Resources, ResourcesRequest,
    ResultDependencySettings, SampleDependencySettings, SampleRequest, SigmaRule,
    SigmaRuleAppliesTo, StageLogsAdd, UserCreate, UserRole, VendorEntityRequest, Volume,
    VolumeTypes, WindowsProcessEntity, WorkerDeleteMap, WorkerRegistrationList,
};
use crate::test_utilities;
use crate::{Error, Thorium};

static UTF8_CHARS: LazyLock<Vec<char>> = LazyLock::new(|| {
    [
        0x0030..=0x0039,   // Numbers
        0x0041..=0x005A,   // Uppercase letters
        0x0061..=0x007A,   // Lowercase letters
        0x2600..=0x26FF,   // Miscellaneous Symbols
        0x1F600..=0x1F64F, // Emoticons
    ]
    .into_iter()
    .flat_map(IntoIterator::into_iter)
    .filter_map(std::char::from_u32)
    .collect()
});

macro_rules! gen_int {
    ($min:expr, $max:expr) => {
        rand::rngs::SmallRng::from_os_rng().random_range($min..$max)
    };
}

/// Generate an option with the given probability that it's `Some`
macro_rules! gen_opt {
    ($prob:literal, $build:expr) => {
        if rand::rngs::SmallRng::from_os_rng().random_bool($prob) {
            Some($build)
        } else {
            None
        }
    };
}

/// generate a random string
pub fn gen_string(len: usize) -> String {
    // build the possible values we can generate
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789-";
    let mut rng = rand::rngs::SmallRng::from_os_rng();
    // generate the correct number of values
    (0..len)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// generate a random string with UTF-8 "characters"
fn gen_utf8_string(num_chars: usize) -> String {
    let mut rng = rand::rngs::SmallRng::from_os_rng();
    // Generate a random string from the selected chars
    (0..num_chars)
        .map(|_| *UTF8_CHARS.choose(&mut rng).unwrap())
        .collect()
}

/// Generate a random group request
#[allow(dead_code)]
#[must_use]
pub fn gen_group() -> GroupRequest {
    let name = gen_string(50);
    GroupRequest::new(name.clone())
        .owners(GroupUsersRequest::default().direct("thorium"))
        .description(format!("{} description", &name))
}

/// Create a number of random groups in Thorium
///
/// # Arguments
///
/// * `cnt` - The number of groups to create
/// * `client` - The client to use when creating these images
#[allow(dead_code)]
pub async fn groups(cnt: usize, client: &Thorium) -> Result<Vec<GroupRequest>, Error> {
    // create a 20 random groups
    let groups: Vec<GroupRequest> = (0..cnt).map(|_| gen_group()).collect();
    // create groups
    for group in &groups {
        client.groups.create(group).await?;
    }
    Ok(groups)
}

/// Create a number of random users in Thorium
///
/// # Arguments
///
/// * `cnt` - The number of users to create
/// * `client` - The client to get a host string from when creating these users
#[allow(dead_code)]
pub async fn users(cnt: usize, client: &Thorium) -> Result<Vec<String>, Error> {
    // generate usernames
    let usernames: Vec<String> = (0..cnt).map(|_| gen_string(24)).collect();
    // generate user creation blueprints
    let blueprints: Vec<UserCreate> = usernames
        .iter()
        .map(|username| {
            // use my
            UserCreate::new(username, gen_string(64), "fake@fake.gov").skip_verification()
        })
        .collect();
    // use default client settings
    let settings = ClientSettings::default();
    // get our secret key
    let secret_key = Some(&test_utilities::CONF.thorium.secret_key);
    // create these users in Thorium
    for bp in blueprints {
        Users::create(&client.host, bp, secret_key, &settings).await?;
    }
    Ok(usernames)
}

#[allow(dead_code)]
pub async fn client(client: &Thorium) -> Result<Thorium, Error> {
    // generate username and password
    let username = gen_string(24);
    let password = gen_string(64);
    // build user create blueprint
    let bp = UserCreate::new(&username, &password, "fake@fake.gov")
        .skip_verification()
        .role(UserRole::Developer {
            k8s: true,
            bare_metal: true,
            windows: true,
            kvm: false,
            external: true,
        });
    // use default client settings
    let settings = ClientSettings::default();
    // get our secret key
    let secret_key = Some(&test_utilities::CONF.thorium.secret_key);
    // create user in Thorium
    Users::create(&client.host, bp, secret_key, &settings).await?;
    // build client for this user
    Thorium::build(&client.host)
        .basic_auth(username, password)
        .build()
        .await
}

/// Generate a random image request
///
/// # Arguments
///
/// * `group` - The group this image should be in
///
/// # Panics
///
/// Panics if semver version fails to parse, which shouldn't happen because
/// we always set it to a good value
#[allow(dead_code)]
#[must_use]
pub fn gen_image(group: &str) -> ImageRequest {
    let name = gen_string(25);
    ImageRequest::new(group, &name)
        .version(ImageVersion::SemVer(
            semver::Version::parse("1.0.0").unwrap(),
        ))
        .image(gen_string(90))
        .lifetime(ImageLifetime::jobs(3))
        .timeout(300)
        .resources(
            ResourcesRequest::default()
                .cores(2.0)
                .memory("1Gi")
                // this will never fail as 1Gi is valid and hardcoded
                .unwrap()
                .nvidia_gpu(1)
                .amd_gpu(5),
        )
        .env("ENV_ARG", "Test")
        .unset_env("REMOVE_ARG")
        .volume(Volume::new("woot", "/woots", VolumeTypes::Secret))
        .description(name + " image description")
        .display_type(OutputDisplayType::String)
        .output_collection(
            OutputCollection::default().files(
                FilesHandler::default()
                    .results("/data/corn")
                    .result_files("/data/corn_files")
                    .names(vec!["corn.png", "corn.json"]),
            ),
        )
        .child_filters(
            ChildFilters::default()
                .mimes([r"(?m)^([^:]+):([0-9]+):(.+)$", r"Hello (?<name>\w+)!"])
                .file_name(r"note.*")
                .file_extension("exe"),
        )
        .clean_up(
            Cleanup::new("/scripts/script.py".to_owned())
                .job_id(ArgStrategy::Kwarg("--job_id".to_owned()))
                .results(ArgStrategy::Kwarg("--results".to_owned()))
                .result_files_dir(ArgStrategy::Append),
        )
        .dependencies(
            Dependencies::default()
                .samples(
                    SampleDependencySettings::default()
                        .location("/test/samples")
                        .kwarg("--samples")
                        .strategy(DependencyPassStrategy::Directory),
                )
                .ephemeral(
                    EphemeralDependencySettings::new("/ephemeral", DependencyPassStrategy::Names)
                        .kwarg("--ephemeral"),
                )
                .results(
                    ResultDependencySettings::new(vec!["plant", "harvest"])
                        .location("/tmp/prior-harvests")
                        .kwarg(KwargDependency::List("--prior".to_owned()))
                        .strategy(DependencyPassStrategy::Names)
                        .name("fields.txt"),
                )
                .repos(
                    RepoDependencySettings::default()
                        .location("/test/repos")
                        .kwarg("--repos")
                        .strategy(DependencyPassStrategy::Directory),
                ),
        )
}

/// Generate a random external image request
///
/// # Arguments
///
/// * `group` - The group this image should be in
#[allow(dead_code)]
#[must_use]
pub fn gen_ext_image(group: &str) -> ImageRequest {
    let name = gen_string(25);
    ImageRequest::new(group, &name)
        .scaler(ImageScaler::External)
        .description(name + " external image description")
}

/// Setup a number of random images in a group
///
/// # Arguments
///
/// * `group` - The group these images should be in
/// * `cnt` - The number of images to create
/// * `client` - The client to use when creating these images
#[allow(dead_code)]
pub async fn images(
    group: &str,
    cnt: usize,
    external: bool,
    client: &Thorium,
) -> Result<Vec<ImageRequest>, Error> {
    // create a 20 random images then
    let images: Vec<ImageRequest> = if external {
        (0..cnt).map(|_| gen_ext_image(group)).collect()
    } else {
        (0..cnt).map(|_| gen_image(group)).collect()
    };
    // create images
    for image in &images {
        client.images.create(image).await?;
    }
    Ok(images)
}

/// Generate an image with a [`crate::models::HostPath`] with the given mount
///
/// # Arguments
///
/// * `group` - The group to create the image in
/// * `path` - The path to set for the `HostPath`
#[allow(dead_code)]
#[must_use]
pub fn gen_host_path<T: Into<String>>(group: &str, path: T) -> ImageRequest {
    gen_image(group).volume(Volume::new(gen_string(20), path, VolumeTypes::HostPath))
}

/// Generate a random pipeline request
///
/// # Arguments
///
/// * `group` - The group this pipeline should be in
/// * `image_cnt` - The number of images in this pipeline
/// * `external` - Whether this pipeline should be built of external images or not
/// * `client` - The client to use when creating the images for this pipeline
#[allow(dead_code)]
pub async fn gen_pipe(
    group: &str,
    image_cnt: usize,
    external: bool,
    client: &Thorium,
) -> Result<PipelineRequest, Error> {
    let pipe_name = gen_string(25);
    // setup random images and get their names
    let images: Vec<String> = images(group, image_cnt, external, client)
        .await?
        .into_iter()
        .map(|image| image.name)
        .collect();
    let order = serde_json::json!(vec![images]);
    let pipe = PipelineRequest::new(group, &pipe_name, order)
        .sla(gen_int!(1, 86400))
        .description(pipe_name + " pipeline description");
    Ok(pipe)
}

/// Generate a random generator pipeline
///
/// # Arguments
///
/// * `group` - The group this pipeline should be in
/// * `image_cnt` - The number of images in this pipeline
/// * `client` - The client to use when creating the images for this pipeline
#[allow(dead_code)]
pub async fn gen_generator_pipe(group: &str, client: &Thorium) -> Result<PipelineRequest, Error> {
    let pipe_name = gen_string(25);
    // build our generator image
    let mut images = vec![gen_image(group).generator()];
    // build our final image
    images.push(gen_image(group));
    // create images
    for image in &images {
        client.images.create(image).await?;
    }
    // get the order of the images to spawn
    let images: Vec<String> = images.into_iter().map(|image| image.name).collect();
    let order = serde_json::json!(images);
    // build a pipeline request
    let pipe = PipelineRequest::new(group, &pipe_name, order)
        .sla(gen_int!(1, 86400))
        .description(pipe_name + " generator pipe description");
    // create this pipeline in Thorium
    client.pipelines.create(&pipe).await?;
    Ok(pipe)
}

/// Setup a number of random pipelines in a group
///
/// # Arguments
///
/// * `group` - The group these pipelines should be in
/// * `cnt` - The number of pipelines to create
/// * `external` - Whether this pipeline should be built of external images or not
/// * `name` - The name of the test that called this
/// * `client` - The client to use when creating these pipelines
#[allow(dead_code)]
pub async fn pipelines(
    group: &str,
    cnt: usize,
    external: bool,
    client: &Thorium,
) -> Result<Vec<PipelineRequest>, Error> {
    // create cnt random pipelines then
    let mut pipelines = Vec::with_capacity(cnt);
    for _ in 0..cnt {
        pipelines.push(gen_pipe(group, 3, external, client).await?);
    }
    // create pipelines
    for pipe in &pipelines {
        client.pipelines.create(pipe).await?;
    }
    Ok(pipelines)
}

/// Generate a simple pipeline with a configurable number of jobs
///
/// This will reuse a pipeline if it already exists.
///
/// # Arguments
///
/// * `group` - The group to create jogs for
/// * `pipeline` - The pipeline to create this jobs for
/// * `reactions` - The number of reactions to create
/// * `thorium` - The client to use when talking to Thorium
#[allow(dead_code)]
pub async fn gen_jobs(
    group: &str,
    pipeline: &PipelineRequest,
    images: &[ImageRequest],
    reactions: u64,
    client: &Thorium,
) -> Result<(), Error> {
    // check if this group exist already
    if client.groups.get(group).await.is_err() {
        // assume the error is because this group doesn't exist yet
        client.groups.create(&GroupRequest::new(group)).await?;
    }
    // crawl the images in this pipeline
    for image in images {
        // check if this image exists already
        if client.images.get(group, &image.name).await.is_err() {
            // assume the error is because this pipeline doesn't exist yet
            client.images.create(image).await?;
        }
    }
    println!(
        "pipelines -> {:#?}",
        client.pipelines.list(group).details().exec().await?.details
    );
    // check if this pipeline already exists
    if client.pipelines.get(group, &pipeline.name).await.is_err() {
        // assume the error is because this pipeline doesn't exist yet
        client.pipelines.create(pipeline).await?;
    }
    // create the reaction request for our job
    let req = ReactionRequest::new(group, &pipeline.name);
    // create a list of the right number of reactions
    let req_list = (0..reactions)
        .map(|_| req.clone())
        .collect::<Vec<ReactionRequest>>();
    // create our reactions if any were added
    if !req_list.is_empty() {
        // create our reactions in bulk
        client.reactions.create_bulk(&req_list).await?;
    }
    Ok(())
}

/// Setup a number of random pipelines and returnn all request data needed
///
/// # Arguments
///
/// * `cnt` - The number of images to create
/// * `name` - The name of the test that called this
/// * `client` - The client to use when creating these images
#[allow(dead_code)]
pub async fn gen_all(
    cnt: usize,
    client: &Thorium,
) -> Result<(Vec<GroupRequest>, Vec<ImageRequest>, Vec<PipelineRequest>), Error> {
    // Create a group
    let groups = groups(cnt, client).await?;
    // build the vectors to store our built images and pipelines
    let mut images: Vec<ImageRequest> = vec![];
    let mut pipes: Vec<PipelineRequest> = vec![];
    // create random images and pipelines
    let mut rng = rand::rngs::SmallRng::from_os_rng();
    for group in &groups {
        let mut group_images: Vec<ImageRequest> = vec![];
        // get cnt internal and external images
        group_images.extend((0..cnt).map(|_| gen_image(&group.name)));
        group_images.extend((0..cnt).map(|_| gen_ext_image(&group.name)));
        // create these images
        for image in &group_images {
            client.images.create(image).await?;
        }
        // create pipelines using these random iamges
        for _ in 0..cnt {
            // generate randome pipeline name
            let pipe_name = gen_string(25);
            // get some random images for this pipeline
            let reqs: Vec<&ImageRequest> = group_images.iter().choose_multiple(&mut rng, cnt);
            let names: Vec<&String> = reqs.iter().map(|item| &item.name).collect();
            // cast the names to a json list
            let order = serde_json::json!(names);
            // create our pipeline request
            let pipe = PipelineRequest::new(&group.name, &pipe_name, order)
                .sla(gen_int!(1, 86400))
                .description(pipe_name + " pipeline description");
            // send our request to the API
            client.pipelines.create(&pipe).await?;
            pipes.push(pipe);
        }
        // add the group images to our full images list
        images.extend(group_images);
    }
    Ok((groups, images, pipes))
}

/// Generate random args for a stage of a reaction
#[allow(dead_code)]
#[must_use]
pub fn gen_args() -> GenericJobArgs {
    // generate a random number of positional args
    let positionals: Vec<String> = (0..gen_int!(3, 10))
        .map(|_| gen_string(gen_int!(5, 64)))
        .collect();
    // generate a random number of positional args
    let kwargs: BTreeMap<String, Vec<String>> = (0..gen_int!(3, 10))
        .map(|_| {
            (
                gen_string(gen_int!(5, 64)),
                vec![gen_string(gen_int!(5, 64))],
            )
        })
        .collect();
    // generate a random number of switches
    let switches: Vec<String> = (0..gen_int!(3, 10))
        .map(|_| gen_string(gen_int!(5, 64)))
        .collect();
    GenericJobArgs::default()
        .positionals(positionals)
        .set_kwargs(kwargs)
        .switches(switches)
}

/// Generate a random [`ReactionRequest`]
///
/// # Arguments
///
/// * `group` - The group this reaction should be in
/// * `pipe` - The pipeline this reaction is for
/// * `tag` - The tag to use for the pipeline
#[allow(dead_code)]
#[must_use]
pub fn gen_reaction(group: &str, pipe: &Pipeline, tag: Option<&str>) -> ReactionRequest {
    // create a reaction request
    let react_req = ReactionRequest::new(group, &pipe.name);
    // inject tags if they exist
    let react_req = match tag {
        Some(tag) => react_req.tag(tag),
        None => react_req,
    };
    // generate and inject args into this reaction request
    pipe.order
        .iter()
        .flatten()
        .fold(react_req, |req, image| req.args(image.clone(), gen_args()))
}

/// Setup a number of random reactions in a group for a specific pipeline
///
/// # Arguments
///
/// * `group` - The group these reactions should be in
/// * `cnt` - The number of reactions to create
/// * `name` - The name of the test that called this
/// * `client` - The client to use when creating these pipelines
#[allow(dead_code)]
pub async fn reactions(
    group: &str,
    cnt: usize,
    tag: Option<&str>,
    client: &Thorium,
) -> Result<(Vec<ReactionRequest>, BulkReactionResponse), Error> {
    // create a random pipeline for these reactions
    let pipe_req = pipelines(group, 1, false, client).await?.remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(group, &pipe_req.name).await?;
    // create reactions requests
    let react_reqs: Vec<ReactionRequest> =
        (0..cnt).map(|_| gen_reaction(group, &pipe, tag)).collect();
    let resp = client.reactions.create_bulk(&react_reqs).await?;
    Ok((react_reqs, resp))
}

/// Creates N random sub reactions
#[allow(dead_code)]
pub async fn sub_reactions(
    group: &str,
    cnt: usize,
    parent: &Uuid,
    client: &Thorium,
) -> Result<(Vec<ReactionRequest>, Vec<ReactionCreation>, Pipeline), Error> {
    // create a random pipeline
    let pipe_req = pipelines(group, 1, false, client).await?.remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(group, &pipe_req.name).await?;
    // track our spawned sub reactions
    let mut sub_reacts = vec![];
    let mut creates = vec![];
    // spawn 3 sub reactions
    for _ in 0..cnt {
        // Create a random reaction based on our pipeline request
        let sub_req = gen_reaction(group, &pipe, None);
        let sub_req = sub_req.parent(*parent);
        // make sure that we were able to create a reaction and our jobs
        let resp = client.reactions.create(&sub_req).await?;
        sub_reacts.push(sub_req);
        creates.push(resp);
    }
    Ok((sub_reacts, creates, pipe))
}

/// Builds random stage logs
///
/// This assumes a return code of 0
#[allow(dead_code)]
#[must_use]
pub fn stage_logs() -> StageLogsAdd {
    // create default stage logs
    let mut logs = StageLogsAdd::default().code(0);
    // create random logs
    let lines = (0..gen_int!(10, 50))
        .map(|_| gen_string(gen_int!(256, 1024)))
        .collect();
    // add random logs
    logs.add_logs(lines);
    logs
}

/// Generate a random sample request
///
/// # Arguments
///
/// * `group` - The group this sample should be in
#[allow(dead_code)]
#[must_use]
pub fn gen_sample(group: &str) -> SampleRequest {
    SampleRequest::new_buffer(Buffer::new(gen_string(gen_int!(2048, 4096))), vec![group])
        .description(gen_string(gen_int!(20, 2048)))
        .tag(gen_string(gen_int!(4, 32)), gen_string(gen_int!(8, 64)))
        .tag(gen_string(gen_int!(4, 32)), gen_string(gen_int!(8, 64)))
        .tag(gen_string(gen_int!(4, 32)), gen_string(gen_int!(8, 64)))
        .origin(OriginRequest::downloaded(
            gen_string(gen_int!(4, 50)),
            Some(gen_string(gen_int!(8, 24))),
        ))
}

/// Setup a number of random samples in a group
///
/// # Arguments
///
/// * `group` - The group these samples should be in
/// * `cnt` - The number of samples to create
/// * `name` - The name of the test that called this
/// * `client` - The client to use when creating these samples
#[allow(dead_code)]
pub async fn samples(
    group: &str,
    cnt: usize,
    client: &Thorium,
) -> Result<Vec<SampleRequest>, Error> {
    // build a sample request
    let reqs = (0..cnt)
        .map(|_| gen_sample(group))
        .collect::<Vec<SampleRequest>>();
    // upload these files
    for req in &reqs {
        client.files.create(req.clone()).await?;
    }
    Ok(reqs)
}

/// Setup a number of random samples in a group that have the same tag
///
/// # Arguments
///
/// * `group` - The group these samples should be in
/// * `cnt` - The number of samples to create
/// * `name` - The name of the test that called this
/// * `client` - The client to use when creating these samples
#[allow(dead_code)]
pub async fn samples_tagged(
    group: &str,
    cnt: usize,
    client: &Thorium,
) -> Result<(String, String, Vec<SampleRequest>), Error> {
    // build a sample request
    let reqs = (0..cnt)
        .map(|_| gen_sample(group))
        .collect::<Vec<SampleRequest>>();
    // build a shared tag for all these requests
    let key = gen_utf8_string(16);
    let value = gen_utf8_string(16);
    // add the same tag to all of our sample requests
    let reqs = reqs
        .into_iter()
        .map(|req| req.tag(&key, &value))
        .collect::<Vec<SampleRequest>>();
    // upload these files
    for req in &reqs {
        client.files.create(req.clone()).await?;
    }
    Ok((key, value, reqs))
}

/// Setup a number of random samples in a group with the given tag
///
/// # Arguments
///
/// * `group` - The group these samples should be in
/// * `cnt` - The number of samples to create
/// * `name` - The name of the test that called this
/// * `client` - The client to use when creating these samples
#[allow(dead_code)]
pub async fn samples_with_tag(
    group: &str,
    cnt: usize,
    key: &str,
    value: &str,
    client: &Thorium,
) -> Result<Vec<SampleRequest>, Error> {
    // build a sample request
    let reqs = (0..cnt)
        .map(|_| gen_sample(group))
        .collect::<Vec<SampleRequest>>();
    // add the same tag to all of our sample requests
    let reqs = reqs
        .into_iter()
        .map(|req| req.tag(key, value))
        .collect::<Vec<SampleRequest>>();
    // upload these files
    for req in &reqs {
        client.files.create(req.clone()).await?;
    }
    Ok(reqs)
}

/// Generate a random repo request
///
/// # Arguments
///
/// * `group` - The group this repo should be in
#[allow(dead_code)]
#[must_use]
pub fn gen_repo(group: &str) -> RepoRequest {
    RepoRequest::new(
        format!(
            "provider.tld/{}/{}",
            gen_string(gen_int!(4, 32)),
            gen_string(gen_int!(4, 32)),
        ),
        vec![group],
        Some(RepoCheckout::branch("main")),
    )
    .tag(gen_string(gen_int!(4, 32)), gen_string(gen_int!(8, 64)))
    .tag(gen_string(gen_int!(4, 32)), gen_string(gen_int!(8, 64)))
    .tag(gen_string(gen_int!(4, 32)), gen_string(gen_int!(8, 64)))
}

/// Setup a number of random repos in a group
///
/// # Arguments
///
/// * `group` - The group these repos should be in
/// * `cnt` - The number of repos to create
/// * `name` - The name of the test that called this
/// * `client` - The client to use when creating these repos
#[allow(dead_code)]
pub async fn repos(group: &str, cnt: usize, client: &Thorium) -> Result<Vec<RepoRequest>, Error> {
    // build a repo request
    let reqs = (0..cnt)
        .map(|_| gen_repo(group))
        .collect::<Vec<RepoRequest>>();
    // upload these repos
    for req in &reqs {
        client.repos.create(req).await?;
    }
    Ok(reqs)
}

/// Setup a number of random repos in a group that have the same tag
///
/// # Arguments
///
/// * `group` - The group these repos should be in
/// * `cnt` - The number of repos to create
/// * `name` - The name of the test that called this
/// * `client` - The client to use when creating these repos
#[allow(dead_code)]
pub async fn repos_tagged(
    group: &str,
    cnt: usize,
    client: &Thorium,
) -> Result<(String, String, Vec<RepoRequest>), Error> {
    // generate a key and value pair
    let key = gen_string(16);
    let value = gen_string(16);
    // build repo requests with those key/value tags
    let reqs = (0..cnt)
        .map(|_| gen_repo(group))
        .map(|req| req.tag(&key, &value))
        .collect::<Vec<RepoRequest>>();
    // upload these repos
    for req in &reqs {
        client.repos.create(req).await?;
    }
    Ok((key, value, reqs))
}

/// Setup a node
///
/// # Arguments
///
/// * `cluster` - The cluster this node should be in
/// * `node` - The name of the node to register
/// * `resources` - The resources this node has
pub async fn node(
    cluster: &str,
    node: &str,
    resources: Resources,
    client: &Thorium,
) -> Result<(), Error> {
    // register this node
    client
        .system
        .register_node(&NodeRegistration::new(cluster, node, resources))
        .await?;
    Ok(())
}

/// Register a worker for a node
///
/// # Arguments
///
/// * `cluster` - The cluster this worker is in
/// * `node` - The node this worker will be on
/// * `name` - The name of this worker
/// * `group` - The group this worker is executing a job in
/// * `pipe` - The pipeline this worker is executing a job for
/// * `stage` - The stage this worker is executing a job for
/// * `client` - The client to register this worker with
pub async fn worker(
    cluster: &str,
    node: &str,
    name: &str,
    group: &str,
    pipe: &str,
    stage: &str,
    client: &Thorium,
) -> Result<(), Error> {
    // get our username
    let user = client.users.info().await?.username;
    // register this worker
    client
        .system
        .register_workers(
            ImageScaler::K8s,
            &WorkerRegistrationList::default().add(
                cluster,
                node,
                name,
                user,
                group,
                pipe,
                stage,
                Resources::default(),
                Pools::Deadline,
            ),
        )
        .await?;
    Ok(())
}

/// Register an external worker for a node
///
/// # Arguments
///
/// * `cluster` - The cluster this worker is in
/// * `node` - The node this worker will be on
/// * `name` - The name of this worker
/// * `group` - The group this worker is executing a job in
/// * `pipe` - The pipeline this worker is executing a job for
/// * `stage` - The stage this worker is executing a job for
/// * `client` - The client to register this worker with
pub async fn worker_ext(
    cluster: &str,
    node: &str,
    name: &str,
    group: &str,
    pipe: &str,
    stage: &str,
    client: &Thorium,
) -> Result<(), Error> {
    // get our username
    let user = client.users.info().await?.username;
    // register this worker
    client
        .system
        .register_workers(
            ImageScaler::External,
            &WorkerRegistrationList::default().add(
                cluster,
                node,
                name,
                user,
                group,
                pipe,
                stage,
                Resources::default(),
                Pools::Deadline,
            ),
        )
        .await?;
    Ok(())
}

/// Delete a worker
///
/// # Arguments
///
/// * `worker` - The name of the worker to delete
/// * `client` - The client to delete this worker with
pub async fn delete_worker(worker: &str, client: &Thorium) -> Result<(), Error> {
    // register this node
    client
        .system
        .delete_workers(ImageScaler::K8s, &WorkerDeleteMap::default().add(worker))
        .await?;
    Ok(())
}

/// Delete an external worker
///
/// # Arguments
///
/// * `worker` - The name of the worker to delete
/// * `client` - The client to delete this worker with
pub async fn delete_worker_ext(worker: &str, client: &Thorium) -> Result<(), Error> {
    // register this node
    client
        .system
        .delete_workers(
            ImageScaler::External,
            &WorkerDeleteMap::default().add(worker),
        )
        .await?;
    Ok(())
}

/// Generate a random [`Ipv4Cidr`]
///
/// Network length is always 24 to keep things simple
#[must_use]
fn gen_ipv4_cidr() -> Ipv4Cidr {
    Ipv4Cidr::new(
        Ipv4Addr::new(gen_int!(1, 255), gen_int!(1, 255), gen_int!(1, 255), 0),
        24,
    )
    .unwrap()
}

/// Generate a random [`Ipv6Cidr`]
///
/// Network length is always 64 to keep things simple
#[must_use]
fn gen_ipv6_cidr() -> Ipv6Cidr {
    Ipv6Cidr::new(
        Ipv6Addr::new(
            gen_int!(1, 65535),
            gen_int!(1, 65535),
            gen_int!(1, 65535),
            gen_int!(1, 65535),
            0,
            0,
            0,
            0,
        ),
        64,
    )
    .unwrap()
}

/// Generate a random [`IpBlock`]
#[must_use]
fn gen_ip_block() -> IpBlock {
    if rand::rngs::SmallRng::from_os_rng().random_bool(0.5) {
        let block = Ipv4Block {
            cidr: gen_ipv4_cidr(),
            // leave "except" as None to avoid issues when checking
            // if it's a subset of the above cidr
            except: None,
        };
        IpBlock::V4(block)
    } else {
        let block = Ipv6Block {
            cidr: gen_ipv6_cidr(),
            // leave "except" as None to avoid issues when checking
            // if it's a subset of the above cidr
            except: None,
        };
        IpBlock::V6(block)
    }
}

#[must_use]
fn gen_custom_network_policy_rule() -> NetworkPolicyCustomK8sRule {
    NetworkPolicyCustomK8sRule {
        // generate a random None/Some list of random custom labels
        namespace_labels: gen_opt!(0.9, {
            (0..10)
                .map(|_| {
                    NetworkPolicyCustomLabel::new(
                        gen_string(gen_int!(1, 63)),
                        gen_string(gen_int!(1, 63)),
                    )
                })
                .collect()
        }),
        pod_labels: gen_opt!(
            0.9,
            (0..10)
                .map(|_| {
                    NetworkPolicyCustomLabel::new(
                        gen_string(gen_int!(1, 63)),
                        gen_string(gen_int!(1, 63)),
                    )
                })
                .collect()
        ),
    }
}

/// Generate random network policy settings
///
/// # Arguments
///
/// * `groups` - The possible groups that will be in the settings
#[must_use]
pub fn gen_network_policy_rule(groups: &[String]) -> NetworkPolicyRuleRaw {
    let allowed_ips = (0..gen_int!(1, 10))
        .map(|_| {
            // create a real ip block to ensure our addresses are valid
            let ip_block = gen_ip_block();
            // convert the ip block to a raw ip block
            let (cidr, except) = match ip_block {
                IpBlock::V4(ipv4_block) => (
                    ipv4_block.cidr.to_string(),
                    ipv4_block
                        .except
                        .map(|except| except.into_iter().map(|cidr| cidr.to_string()).collect()),
                ),
                IpBlock::V6(ipv6_block) => (
                    ipv6_block.cidr.to_string(),
                    ipv6_block
                        .except
                        .map(|except| except.into_iter().map(|cidr| cidr.to_string()).collect()),
                ),
            };
            IpBlockRaw { cidr, except }
        })
        .collect();
    let ports = (0..gen_int!(1, 10))
        .map(|_| NetworkPolicyPort {
            port: gen_int!(1, 65535),
            end_port: gen_opt!(0.5, gen_int!(1, 65535)),
            protocol: gen_opt!(0.5, {
                if rand::rngs::SmallRng::from_os_rng().random_bool(0.5) {
                    NetworkProtocol::TCP
                } else {
                    NetworkProtocol::UDP
                }
            }),
        })
        .collect();
    let allowed_custom = (0..gen_int!(0, 10))
        .map(|_| gen_custom_network_policy_rule())
        .collect();
    NetworkPolicyRuleRaw {
        allowed_ips,
        allowed_groups: groups
            .choose_multiple(
                &mut rand::rngs::SmallRng::from_os_rng(),
                gen_int!(0, groups.len()),
            )
            .cloned()
            .collect(),
        // refrain from adding allowed tools to avoid failing tools existing check
        allowed_tools: Vec::new(),
        allowed_local: rand::rngs::SmallRng::from_os_rng().random_bool(0.5),
        allowed_internet: rand::rngs::SmallRng::from_os_rng().random_bool(0.5),
        allowed_all: rand::rngs::SmallRng::from_os_rng().random_bool(0.5),
        ports,
        allowed_custom,
    }
}

/// Generate a network policy request
///
/// # Arguments
///
/// * `groups` - The groups the network policy will be in
#[must_use]
pub fn gen_network_policy(groups: &[String]) -> NetworkPolicyRequest {
    // generate a random number of ingress/egress rules
    let ingress = gen_opt!(
        0.9,
        (0..gen_int!(1, 10))
            .map(|_| gen_network_policy_rule(groups))
            .collect()
    );
    let egress = gen_opt!(
        0.9,
        (0..gen_int!(1, 10))
            .map(|_| gen_network_policy_rule(groups))
            .collect()
    );
    NetworkPolicyRequest {
        name: gen_utf8_string(gen_int!(1, 63)),
        groups: groups.to_vec(),
        ingress,
        egress,
        forced_policy: false,
        default_policy: false,
    }
}

/// Create the given number of network policies in Thorium
///
/// # Arguments
///
/// * `groups` - The groups the network policies should be in
/// * `cnt` - The number of network policies to create
/// * `client` - The Thorium client
pub async fn network_policies(
    groups: &[String],
    cnt: usize,
    client: &Thorium,
) -> Result<Vec<NetworkPolicyRequest>, Error> {
    // generate the requests
    let reqs: Vec<NetworkPolicyRequest> = (0..cnt).map(|_| gen_network_policy(groups)).collect();
    // create the network policies concurrently
    stream::iter(reqs.iter())
        .map(Ok::<&NetworkPolicyRequest, Error>)
        .try_for_each_concurrent(100, |req| async {
            client.network_policies.create(req.clone()).await?;
            Ok(())
        })
        .await?;
    // return the created requests
    Ok(reqs)
}

// Entity generators

/// A minimal, valid sigma rule used to generate sigma rule entities in tests
const TEST_SIGMA_RULE: &str = r#"title: A rule with keywords
logsource:
    service: test
detection:
    keywords:
        - '* hello world?'
        - 'evil'
    condition: keywords
"#;

/// Generate metadata for an entity with no unique kind (an `Other` entity)
#[allow(dead_code)]
#[must_use]
pub fn gen_other_meta() -> EntityMetadataRequest {
    // other entities have no metadata
    EntityMetadataRequest::Other
}

/// Generate metadata for a random vendor entity
#[allow(dead_code)]
#[must_use]
pub fn gen_vendor_meta() -> EntityMetadataRequest {
    // build a set of countries this vendor operates in (alpha-2 codes)
    let mut countries = BTreeSet::new();
    countries.insert("US".to_owned());
    // build the critical sectors this vendor is associated with
    let mut critical_sectors = BTreeSet::new();
    critical_sectors.insert(CriticalSector::InformationTechnology);
    // build our vendor metadata request
    EntityMetadataRequest::Vendor(VendorEntityRequest {
        countries,
        critical_sectors,
    })
}

/// Generate metadata for a random device entity
///
/// # Arguments
///
/// * `vendors` - The ids of the vendor entities this device is associated with
#[allow(dead_code)]
#[must_use]
pub fn gen_device_meta(vendors: Vec<Uuid>) -> EntityMetadataRequest {
    // build the critical sectors this device is associated with
    let mut critical_sectors = BTreeSet::new();
    critical_sectors.insert(CriticalSector::Energy);
    // build our device metadata request
    EntityMetadataRequest::Device(DeviceEntityRequest {
        urls: vec![format!("https://{}.example.com", gen_string(8))],
        vendors,
        critical_system: Some(true),
        sensitive_location: Some(false),
        critical_sectors,
    })
}

/// Generate metadata for a random flag entity
#[allow(dead_code)]
#[must_use]
pub fn gen_flag_meta() -> EntityMetadataRequest {
    // build our flag metadata request
    EntityMetadataRequest::Flag(Flag {
        suspicion: gen_int!(0, 100),
        confidence: Confidence::Likely,
        content: Some(gen_string(gen_int!(8, 32))),
        reasoning: gen_string(gen_int!(8, 64)),
    })
}

/// Generate metadata for a sigma rule entity
#[allow(dead_code)]
#[must_use]
pub fn gen_sigma_meta() -> EntityMetadataRequest {
    // build a validated sigma rule from our test rule
    let rule = SigmaRule::new(TEST_SIGMA_RULE, SigmaRuleAppliesTo::WindowsProcesses)
        .expect("failed to build test sigma rule");
    // build our sigma rule metadata request
    EntityMetadataRequest::SigmaRule(rule)
}

/// Generate metadata for a random network connection entity
#[allow(dead_code)]
#[must_use]
pub fn gen_network_connection_meta() -> EntityMetadataRequest {
    // build our network connection metadata request, leaving timestamps unset
    EntityMetadataRequest::NetworkConnection(NetworkConnection {
        protocol: Some(TransportLayerProtocol::TCP),
        source: IpAddr::V4(Ipv4Addr::new(10, 0, 0, gen_int!(1, 254))),
        source_port: Some(gen_int!(1024, 65535)),
        destination: IpAddr::V4(Ipv4Addr::new(10, 0, 0, gen_int!(1, 254))),
        destination_port: gen_int!(1, 65535),
        state: Some(NetConState::Established),
        pid: Some(gen_int!(1, 65535)),
        process: Some(gen_string(gen_int!(4, 16))),
        create_time: None,
    })
}

/// Generate metadata for a random collection entity
#[allow(dead_code)]
#[must_use]
pub fn gen_collection_meta() -> EntityMetadataRequest {
    // build a single random tag for this collection
    let mut collection_tags = BTreeMap::new();
    let mut values = BTreeSet::new();
    values.insert(gen_string(gen_int!(4, 16)));
    collection_tags.insert(gen_string(gen_int!(4, 16)), values);
    // build our collection metadata request, leaving timestamps unset
    EntityMetadataRequest::Collection(CollectionEntityRequest {
        collection_kind: CollectionKind::Files,
        collection_tags,
        tags_case_insensitive: Some(false),
        ignore_groups: Some(false),
        start: None,
        end: None,
    })
}

/// Generate metadata for a random filesystem entity
#[allow(dead_code)]
#[must_use]
pub fn gen_filesystem_meta() -> EntityMetadataRequest {
    // build our filesystem metadata request
    EntityMetadataRequest::FileSystem(FileSystemEntity {
        sha256: gen_string(64),
        tools: vec![gen_string(gen_int!(4, 16))],
    })
}

/// Generate metadata for a random filesystem folder entity
///
/// # Arguments
///
/// * `filesystem_id` - The id of the filesystem entity this folder belongs to
#[allow(dead_code)]
#[must_use]
pub fn gen_folder_meta(filesystem_id: Uuid) -> EntityMetadataRequest {
    // build our folder metadata request
    EntityMetadataRequest::Folder(FileSystemFolderEntity {
        filesystem_id,
        names_sha256: gen_string(64),
        data_sha256: gen_string(64),
        all_sha256: gen_string(64),
    })
}

/// Generate metadata for a windows process tree entity
#[allow(dead_code)]
#[must_use]
pub fn gen_windows_process_tree_meta() -> EntityMetadataRequest {
    // windows process tree entities have no metadata
    EntityMetadataRequest::WindowsProcessTree
}

/// Generate metadata for a random windows process entity
#[allow(dead_code)]
#[must_use]
pub fn gen_windows_process_meta() -> EntityMetadataRequest {
    // build a process with only a pid and a few descriptive fields set
    let mut process = WindowsProcessEntity::new(gen_int!(1, 65535));
    process.name = Some(gen_string(gen_int!(4, 16)));
    process.image_path = Some(format!("C:\\\\{}.exe", gen_string(gen_int!(4, 16))));
    process.command = Some(gen_string(gen_int!(8, 32)));
    process.threads = Some(gen_int!(1, 64));
    process.handles = Some(gen_int!(1, 256));
    process.session_id = Some(gen_int!(0, 8));
    // build our windows process metadata request
    EntityMetadataRequest::WindowsProcess(process)
}

/// Generate metadata for a random PE section entity
#[allow(dead_code)]
#[must_use]
pub fn gen_pe_section_meta() -> EntityMetadataRequest {
    // build our PE section metadata request with a clean entropy value
    EntityMetadataRequest::PeSection(PeSectionEntity {
        md5: Some(gen_string(32)),
        raw_size: Some(gen_int!(1, 100_000)),
        virtual_size: Some(gen_int!(1, 100_000)),
        entropy: Some(7.5),
    })
}

/// Generate metadata for a random PE import entity
#[allow(dead_code)]
#[must_use]
pub fn gen_pe_import_meta() -> EntityMetadataRequest {
    // build our PE import metadata request with a couple of imported functions
    EntityMetadataRequest::PeImport(
        PeImportEntity::new()
            .function(gen_string(gen_int!(4, 16)))
            .function(gen_string(gen_int!(4, 16))),
    )
}

/// Generate metadata for a random incident entity
#[allow(dead_code)]
#[must_use]
pub fn gen_incident_meta() -> EntityMetadataRequest {
    // build our incident metadata request with a cover term and a few list fields
    EntityMetadataRequest::Incident(IncidentRequest {
        cover_term: Some(gen_string(gen_int!(4, 16))),
        mission_teams: vec![gen_string(gen_int!(4, 16)), gen_string(gen_int!(4, 16))],
        networks: vec![gen_string(gen_int!(4, 16))],
        machines: vec![gen_string(gen_int!(4, 16))],
        locations: vec![gen_string(gen_int!(4, 16))],
    })
}

/// Generate metadata for a random compiled function entity
#[allow(dead_code)]
#[must_use]
pub fn gen_compiled_function_meta() -> EntityMetadataRequest {
    // build a couple of disassembled instructions for this function
    let disassembly = vec![
        CompiledInstruction {
            address: gen_int!(1, 100_000),
            instruction: gen_string(gen_int!(4, 16)),
        },
        CompiledInstruction {
            address: gen_int!(1, 100_000),
            instruction: gen_string(gen_int!(4, 16)),
        },
    ];
    // build our compiled function metadata request
    EntityMetadataRequest::CompiledFunction(CompiledFunction {
        address: gen_int!(1, 100_000),
        disassembly,
    })
}

/// Generate metadata for a random decompiled function entity
#[allow(dead_code)]
#[must_use]
pub fn gen_decompiled_function_meta() -> EntityMetadataRequest {
    // build our decompiled function metadata request
    EntityMetadataRequest::DecompiledFunction(DecompiledFunction {
        address: gen_int!(1, 100_000),
        tools: vec![gen_string(gen_int!(4, 16))],
        content: gen_string(gen_int!(16, 128)),
    })
}

/// A fixed second-precision timestamp for update tests
///
/// Whole-second timestamps round-trip cleanly through scylla's millisecond
/// precision, avoiding spurious mismatches when comparing datetimes.
///
/// # Arguments
///
/// * `offset_secs` - The number of seconds to offset from the known base epoch second
#[allow(dead_code)]
#[must_use]
fn fixed_timestamp(offset_secs: i64) -> chrono::DateTime<chrono::Utc> {
    // build a fixed timestamp offset from a known epoch second
    chrono::DateTime::from_timestamp(1_700_000_000 + offset_secs, 0)
        .expect("failed to build fixed timestamp")
}

/// Generate a device metadata update touching every device-specific field
///
/// # Arguments
///
/// * `existing` - The created entity to pull removal targets from
#[allow(dead_code)]
#[must_use]
pub fn gen_device_update(existing: &Entity) -> EntityMetadataUpdate {
    // pull an existing url and critical sector to remove
    let (remove_urls, remove_critical_sectors) = match &existing.metadata {
        EntityMetadata::Device(dev) => (
            dev.urls.first().cloned().into_iter().collect(),
            dev.critical_sectors.iter().next().copied().into_iter().collect(),
        ),
        _ => (Vec::new(), Vec::new()),
    };
    // build a device update that adds/removes urls and sectors and toggles flags
    EntityMetadataUpdate::Device {
        add_urls: vec![format!("https://{}.example.com", gen_string(8))],
        remove_urls,
        critical_system: Some(false),
        clear_critical_system: None,
        sensitive_location: None,
        clear_sensitive_location: Some(true),
        add_critical_sectors: vec![CriticalSector::Communications],
        remove_critical_sectors,
    }
}

/// Generate a vendor metadata update touching every vendor-specific field
#[allow(dead_code)]
#[must_use]
pub fn gen_vendor_update(_existing: &Entity) -> EntityMetadataUpdate {
    // the create generator always uses US and InformationTechnology
    EntityMetadataUpdate::Vendor {
        add_countries: vec!["CA".to_owned()],
        remove_countries: vec!["US".to_owned()],
        add_critical_sectors: vec![CriticalSector::Communications],
        remove_critical_sectors: vec![CriticalSector::InformationTechnology],
    }
}

/// Generate a collection metadata update touching every collection-specific field
///
/// # Arguments
///
/// * `existing` - The created entity to pull removal targets from
#[allow(dead_code)]
#[must_use]
pub fn gen_collection_update(existing: &Entity) -> EntityMetadataUpdate {
    // pull an existing tag to delete
    let delete_collection_tags = match &existing.metadata {
        EntityMetadata::Collection(col) => col
            .collection_tags
            .iter()
            .next()
            .map(|(key, values)| {
                let mut map = std::collections::HashMap::new();
                map.insert(key.clone(), values.iter().cloned().collect());
                map
            })
            .unwrap_or_default(),
        _ => std::collections::HashMap::new(),
    };
    // add a new random tag
    let mut add_collection_tags = std::collections::HashMap::new();
    let mut values = std::collections::HashSet::new();
    values.insert(gen_string(gen_int!(4, 16)));
    add_collection_tags.insert(gen_string(gen_int!(4, 16)), values);
    // build our collection update
    EntityMetadataUpdate::Collection {
        add_collection_tags,
        delete_collection_tags,
        tags_case_insensitive: Some(true),
        ignore_groups: Some(true),
        // the api requires the start to be more recent than the end
        start: Some(fixed_timestamp(100)),
        end: Some(fixed_timestamp(0)),
        clear_start: None,
        clear_end: None,
    }
}

/// Generate a filesystem metadata update touching every filesystem-specific field
///
/// # Arguments
///
/// * `existing` - The created entity to pull removal targets from
#[allow(dead_code)]
#[must_use]
pub fn gen_filesystem_update(existing: &Entity) -> EntityMetadataUpdate {
    // pull an existing tool to remove
    let remove_tools = match &existing.metadata {
        EntityMetadata::FileSystem(fs) => fs.tools.first().cloned().into_iter().collect(),
        _ => Vec::new(),
    };
    // build a filesystem update that adds and removes a tool
    EntityMetadataUpdate::FileSystem {
        add_tools: vec![gen_string(gen_int!(4, 16))],
        remove_tools,
    }
}

/// Generate a windows process tree metadata update
///
/// Process trees start with no tools, so this only exercises adding a tool.
#[allow(dead_code)]
#[must_use]
pub fn gen_windows_process_tree_update(_existing: &Entity) -> EntityMetadataUpdate {
    // build a process tree update that adds a tool
    EntityMetadataUpdate::WindowsProcessTree {
        add_tools: vec![gen_string(gen_int!(4, 16))],
        remove_tools: Vec::new(),
    }
}

/// Generate a windows process metadata update touching every process-specific field
#[allow(dead_code)]
#[must_use]
pub fn gen_windows_process_update(_existing: &Entity) -> EntityMetadataUpdate {
    // build a process update that sets every scalar field
    EntityMetadataUpdate::WindowsProcess {
        name: Some(gen_string(gen_int!(4, 16))),
        image_path: Some(format!("C:\\\\{}.exe", gen_string(gen_int!(4, 16)))),
        command: Some(gen_string(gen_int!(8, 32))),
        offset: Some(gen_int!(1, 100_000)),
        threads: Some(gen_int!(1, 64)),
        handles: Some(gen_int!(1, 256)),
        is_wow64: Some(true),
        session_id: Some(gen_int!(0, 8)),
        create_time: Some(fixed_timestamp(0)),
        exit_time: Some(fixed_timestamp(100)),
    }
}

/// Generate a network connection metadata update touching every field
#[allow(dead_code)]
#[must_use]
pub fn gen_network_connection_update(_existing: &Entity) -> EntityMetadataUpdate {
    // build a network connection update that sets every field
    EntityMetadataUpdate::NetworkConnection {
        protocol: Some(TransportLayerProtocol::UDP),
        source: Some(IpAddr::V4(Ipv4Addr::new(192, 168, 1, gen_int!(1, 254)))),
        source_port: Some(gen_int!(1024, 65535)),
        destination: Some(IpAddr::V4(Ipv4Addr::new(192, 168, 1, gen_int!(1, 254)))),
        destination_port: Some(gen_int!(1, 65535)),
        state: Some(NetConState::Closed),
        pid: Some(gen_int!(1, 65535)),
        process: Some(gen_string(gen_int!(4, 16))),
        create_time: Some(fixed_timestamp(0)),
    }
}

/// Generate a sigma rule metadata update touching every sigma-specific field
#[allow(dead_code)]
#[must_use]
pub fn gen_sigma_update(_existing: &Entity) -> EntityMetadataUpdate {
    // build a sigma rule update that changes the rule, score, applies-to, and actions
    EntityMetadataUpdate::SigmaRule {
        sigma_rule: Some(TEST_SIGMA_RULE.to_owned()),
        score: Some(gen_int!(1, 100)),
        add_applies_to: vec![SigmaRuleAppliesTo::NetworkConnections],
        remove_applies_to: vec![SigmaRuleAppliesTo::WindowsProcesses],
        add_actions: vec![SigmaActionToTake::Flag(SigmaAutoFlag {
            confidence: Confidence::Likely,
            content: Some(gen_string(gen_int!(4, 16))),
            reasoning: gen_string(gen_int!(8, 32)),
        })],
        remove_actions: std::collections::BTreeSet::new(),
    }
}

/// Generate a flag metadata update touching every flag-specific field
#[allow(dead_code)]
#[must_use]
pub fn gen_flag_update(_existing: &Entity) -> EntityMetadataUpdate {
    // build a flag update that sets every field
    EntityMetadataUpdate::Flag {
        suspicion: Some(gen_int!(0, 100)),
        confidence: Some(Confidence::Unsure),
        reasoning: Some(gen_string(gen_int!(8, 32))),
        content: Some(gen_string(gen_int!(4, 16))),
    }
}

/// Generate an incident metadata update touching every incident-specific field
///
/// # Arguments
///
/// * `existing` - The created entity to pull removal targets from
#[allow(dead_code)]
#[must_use]
pub fn gen_incident_update(existing: &Entity) -> EntityMetadataUpdate {
    // pull one existing value from each list to remove
    let (remove_mission_teams, remove_networks, remove_machines, remove_locations) =
        match &existing.metadata {
            EntityMetadata::Incident(incident) => (
                incident.mission_teams.first().cloned().into_iter().collect(),
                incident.networks.first().cloned().into_iter().collect(),
                incident.machines.first().cloned().into_iter().collect(),
                incident.locations.first().cloned().into_iter().collect(),
            ),
            _ => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        };
    // build an incident update that sets the cover term and edits every list
    EntityMetadataUpdate::Incident {
        cover_term: Some(gen_string(gen_int!(4, 16))),
        add_mission_teams: vec![gen_string(gen_int!(4, 16))],
        remove_mission_teams,
        add_networks: vec![gen_string(gen_int!(4, 16))],
        remove_networks,
        add_machines: vec![gen_string(gen_int!(4, 16))],
        remove_machines,
        add_locations: vec![gen_string(gen_int!(4, 16))],
        remove_locations,
    }
}

/// Generate a compiled function metadata update touching every field
#[allow(dead_code)]
#[must_use]
pub fn gen_compiled_function_update(_existing: &Entity) -> EntityMetadataUpdate {
    // build a compiled function update that sets the address and replaces disassembly
    EntityMetadataUpdate::CompiledFunction {
        function_address: Some(gen_int!(1, 100_000)),
        disassembly: vec![CompiledInstruction {
            address: gen_int!(1, 100_000),
            instruction: gen_string(gen_int!(4, 16)),
        }],
    }
}

/// Generate a decompiled function metadata update touching every field
///
/// # Arguments
///
/// * `existing` - The created entity to pull removal targets from
#[allow(dead_code)]
#[must_use]
pub fn gen_decompiled_function_update(existing: &Entity) -> EntityMetadataUpdate {
    // pull an existing tool to remove
    let remove_tools = match &existing.metadata {
        EntityMetadata::DecompiledFunction(decomp) => {
            decomp.tools.first().cloned().into_iter().collect()
        }
        _ => Vec::new(),
    };
    // build a decompiled function update that sets the address, content, and tools
    EntityMetadataUpdate::DecompiledFunction {
        function_address: Some(gen_int!(1, 100_000)),
        decompilation_content: Some(gen_string(gen_int!(16, 128))),
        add_tools: vec![gen_string(gen_int!(4, 16))],
        remove_tools,
    }
}

/// Generate a PE section metadata update touching every field
#[allow(dead_code)]
#[must_use]
pub fn gen_pe_section_update(_existing: &Entity) -> EntityMetadataUpdate {
    // build a PE section update that sets every field
    EntityMetadataUpdate::PeSection {
        md5: Some(gen_string(32)),
        raw_size: Some(gen_int!(1, 100_000)),
        virtual_size: Some(gen_int!(1, 100_000)),
        entropy: Some(3.25),
    }
}

/// Generate a PE import metadata update replacing the imported functions
#[allow(dead_code)]
#[must_use]
pub fn gen_pe_import_update(_existing: &Entity) -> EntityMetadataUpdate {
    // build a PE import update that replaces the imported functions
    EntityMetadataUpdate::PeImport {
        functions: vec![gen_string(gen_int!(4, 16)), gen_string(gen_int!(4, 16))],
    }
}

/// Build an entity request with a random name, description, and tags
///
/// # Arguments
///
/// * `group` - The group this entity should be in
/// * `metadata` - The kind-specific metadata for this entity
#[allow(dead_code)]
#[must_use]
pub fn gen_entity(group: &str, metadata: EntityMetadataRequest) -> EntityRequest {
    // build our entity request with a random name and two random tags
    let mut req = EntityRequest::new(gen_string(gen_int!(8, 32)), metadata, vec![group.to_owned()])
        .tag(gen_string(gen_int!(4, 16)), gen_string(gen_int!(4, 16)))
        .tag(gen_string(gen_int!(4, 16)), gen_string(gen_int!(4, 16)));
    // add a random description
    req.description = Some(gen_string(gen_int!(8, 64)));
    req
}

/// Create an entity in Thorium and return both the request and the created entity
///
/// # Arguments
///
/// * `req` - The entity request to create
/// * `client` - The client to use to create the entity
#[allow(dead_code)]
pub async fn entity(
    req: EntityRequest,
    client: &Thorium,
) -> Result<(EntityRequest, Entity), Error> {
    // create the entity in Thorium
    let resp = client.entities.create(req.clone()).await?;
    // get the full entity we just created
    let entity = client.entities.get(resp.id).await?;
    Ok((req, entity))
}

/// Create a vendor entity in a group and return the created entity
///
/// This is used to satisfy the vendor dependency of device entities.
///
/// # Arguments
///
/// * `group` - The group this vendor should be in
/// * `client` - The client to use to create the vendor
#[allow(dead_code)]
pub async fn vendor_entity(group: &str, client: &Thorium) -> Result<Entity, Error> {
    // build and create a vendor entity
    let (_, entity) = entity(gen_entity(group, gen_vendor_meta()), client).await?;
    Ok(entity)
}

/// Create a filesystem entity in a group and return the created entity
///
/// This is used to satisfy the filesystem dependency of folder entities.
///
/// # Arguments
///
/// * `group` - The group this filesystem should be in
/// * `client` - The client to use to create the filesystem
#[allow(dead_code)]
pub async fn filesystem_entity(group: &str, client: &Thorium) -> Result<Entity, Error> {
    // build and create a filesystem entity
    let (_, entity) = entity(gen_entity(group, gen_filesystem_meta()), client).await?;
    Ok(entity)
}

// Generators for sync tests

#[cfg(all(feature = "sync", not(feature = "python")))]
use crate::ThoriumBlocking;

/// Create a number of random groups in Thorium
///
/// # Arguments
///
/// * `cnt` - The number of groups to create
/// * `client` - The client to use when creating these images
#[cfg(all(feature = "sync", not(feature = "python")))]
#[allow(dead_code)]
pub fn groups_blocking(cnt: usize, client: &ThoriumBlocking) -> Result<Vec<GroupRequest>, Error> {
    // create a 20 random groups
    let groups: Vec<GroupRequest> = (0..cnt).map(|_| gen_group()).collect();
    // create groups
    for group in &groups {
        client.groups.create(group)?;
    }
    Ok(groups)
}

/// Setup a number of random images in a group
///
/// # Arguments
///
/// * `group` - The group these images should be in
/// * `cnt` - The number of images to create
/// * `client` - The client to use when creating these images
#[cfg(all(feature = "sync", not(feature = "python")))]
#[allow(dead_code)]
pub fn images_blocking(
    group: &str,
    cnt: usize,
    external: bool,
    client: &ThoriumBlocking,
) -> Result<Vec<ImageRequest>, Error> {
    // create a 20 random images then
    let images: Vec<ImageRequest> = if external {
        (0..cnt).map(|_| gen_ext_image(group)).collect()
    } else {
        (0..cnt).map(|_| gen_image(group)).collect()
    };
    // create images
    for image in &images {
        client.images.create(image)?;
    }
    Ok(images)
}
