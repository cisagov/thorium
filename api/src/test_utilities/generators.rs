use cidr::{Ipv4Cidr, Ipv6Cidr};
use futures::{stream, StreamExt, TryStreamExt};
use rand::seq::IndexedRandom;
use rand::{seq::IteratorRandom, Rng, SeedableRng};
use std::collections::HashMap;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::LazyLock;
use uuid::Uuid;

use crate::client::{ClientSettings, Users};
use crate::models::{
    ArgStrategy, Buffer, BulkReactionResponse, ChildFilters, Cleanup, Dependencies,
    DependencyPassStrategy, EphemeralDependencySettings, FilesHandler, GenericJobArgs,
    GroupRequest, GroupUsersRequest, ImageLifetime, ImageRequest, ImageScaler, ImageVersion,
    IpBlock, IpBlockRaw, Ipv4Block, Ipv6Block, KwargDependency, NetworkPolicyCustomK8sRule,
    NetworkPolicyCustomLabel, NetworkPolicyPort, NetworkPolicyRequest, NetworkPolicyRuleRaw,
    NetworkProtocol, NodeRegistration, OriginRequest, OutputCollection, OutputDisplayType,
    Pipeline, PipelineRequest, Pools, ReactionCreation, ReactionRequest, RepoCheckout,
    RepoDependencySettings, RepoRequest, Resources, ResourcesRequest, ResultDependencySettings,
    SampleDependencySettings, SampleRequest, StageLogsAdd, UserCreate, UserRole, Volume,
    VolumeTypes, WorkerDeleteMap, WorkerRegistrationList,
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
fn gen_string(len: usize) -> String {
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
/// * ``
#[allow(dead_code)]
pub async fn users(cnt: usize, client: &Thorium) -> Result<Vec<String>, Error> {
    // generate usernames
    let usernames: Vec<String> = (0..cnt).map(|_| gen_string(24)).collect();
    // generate user creation blueprints
    let blueprints: Vec<UserCreate> = usernames
        .iter()
        .map(|username| {
            // use my
            UserCreate::new(username, &gen_string(64), "fake@fake.gov").skip_verification()
        })
        .collect();
    // use default client settings
    let settings = ClientSettings::default();
    // get our secret key
    let secret_key = Some(&test_utilities::config_ref().thorium.secret_key);
    // create these users in Thorium
    for bp in blueprints.into_iter() {
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
    let secret_key = Some(&test_utilities::config_ref().thorium.secret_key);
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
#[allow(dead_code)]
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
pub fn gen_ext_image(group: &str) -> ImageRequest {
    let name = gen_string(25);
    ImageRequest::new(group, &name)
        .scaler(ImageScaler::External)
        .description(name + " external image description")
}

/// Set up a number of random images in a group
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
    let images: Vec<ImageRequest> = if !external {
        (0..cnt).map(|_| gen_image(group)).collect()
    } else {
        (0..cnt).map(|_| gen_ext_image(group)).collect()
    };
    // create images
    for image in images.iter() {
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
    for image in images.iter() {
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

/// Set up a number of random pipelines in a group
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
    for pipe in pipelines.iter() {
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
    for image in images.iter() {
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

/// Set up a number of random pipelines and return all request data needed
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
    for group in groups.iter() {
        let mut group_images: Vec<ImageRequest> = vec![];
        // get cnt internal and external images
        group_images.extend((0..cnt).map(|_| gen_image(&group.name)));
        group_images.extend((0..cnt).map(|_| gen_ext_image(&group.name)));
        // create these images
        for image in group_images.iter() {
            client.images.create(image).await?;
        }
        // create pipelines using these random images
        for _ in 0..cnt {
            // generate random pipeline name
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
pub fn gen_args() -> GenericJobArgs {
    // generate a random number of positional args
    let positionals: Vec<String> = (0..gen_int!(3, 10))
        .map(|_| gen_string(gen_int!(5, 64)))
        .collect();
    // generate a random number of positional args
    let kwargs: HashMap<String, Vec<String>> = (0..gen_int!(3, 10))
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
pub fn gen_reaction(group: &str, pipe: &Pipeline, tag: Option<&str>) -> ReactionRequest {
    // create a reaction request
    let react_req = ReactionRequest::new(group, &pipe.name);
    // inject tags if they exist
    let react_req = match tag {
        Some(tag) => react_req.tag(tag),
        None => react_req,
    };
    // generate and inject args into this reaction request
    let react_req = pipe
        .order
        .iter()
        .flatten()
        .fold(react_req, |req, image| req.args(image.clone(), gen_args()));
    react_req
}

/// Set up a number of random reactions in a group for a specific pipeline
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
    let pipe_req = pipelines(&group, 1, false, client).await?.remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // track our spawned sub reactions
    let mut sub_reacts = vec![];
    let mut creates = vec![];
    // spawn 3 sub reactions
    for _ in 0..cnt {
        // Create a random reaction based on our pipeline request
        let sub_req = gen_reaction(&group, &pipe, None);
        let sub_req = sub_req.parent(parent.clone());
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

/// Set up a number of random samples in a group
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
    for req in reqs.iter() {
        client.files.create(req.clone()).await?;
    }
    Ok(reqs)
}

/// Set up a number of random samples in a group that have the same tag
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

/// Set up a number of random repos in a group
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

/// Set up a number of random repos in a group that have the same tag
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

/// Set up a node
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
