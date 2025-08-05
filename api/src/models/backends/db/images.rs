use bb8_redis::redis::cmd;
use std::collections::HashMap;
use tracing::instrument;
use uuid::Uuid;

use super::helpers;
use super::keys::{GroupKeys, ImageKeys, SystemKeys};
use crate::models::backends::NotificationSupport;
use crate::models::{
    Group, Image, ImageBan, ImageJobInfo, ImageKey, ImageList, ImageRequest, ImageScaler, User,
};
use crate::utils::{ApiError, Shared};
use crate::{
    cast, coerce_bool, conflict, conn, deserialize, exec_query, hset_del_opt_serialize,
    hsetnx_opt_serialize, not_found, query, serialize,
};

/// Builds a image creation pipeline for Redis
///
/// # Arguments
///
/// * `pipe` - The redis pipeline to add onto
/// * `cast` - The user to create in redis
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub fn build(
    pipe: &mut redis::Pipeline,
    cast: &Image,
    shared: &Shared,
) -> Result<(), ApiError> {
    // build image keys
    let keys = ImageKeys::new(cast, shared);
    // build key to system info
    let syskey = SystemKeys::new(shared);
    // add command to pipeline
    pipe.cmd("hsetnx").arg(&keys.data).arg("group").arg(&cast.group)
        .cmd("hsetnx").arg(&keys.data).arg("name").arg(&cast.name)
        .cmd("hsetnx").arg(&keys.data).arg("creator").arg(&cast.creator)
        .cmd("hsetnx").arg(&keys.data).arg("scaler").arg(serialize!(&cast.scaler))
        .cmd("hsetnx").arg(&keys.data).arg("resources").arg(serialize!(&cast.resources))
        .cmd("hsetnx").arg(&keys.data).arg("spawn_limit").arg(serialize!(&cast.spawn_limit))
        .cmd("hsetnx").arg(&keys.data).arg("runtime").arg(cast.runtime)
        .cmd("hsetnx").arg(&keys.data).arg("volumes").arg(serialize!(&cast.volumes))
        .cmd("hsetnx").arg(&keys.data).arg("env").arg(serialize!(&cast.env))
        .cmd("hsetnx").arg(&keys.data).arg("args").arg(serialize!(&cast.args))
        .cmd("hsetnx").arg(&keys.data).arg("security_context")
            .arg(serialize!(&cast.security_context))
        .cmd("hsetnx").arg(&keys.data).arg("collect_logs")
            .arg(serialize!(&cast.collect_logs))
        .cmd("hsetnx").arg(&keys.data).arg("generator").arg(serialize!(&cast.generator))
        .cmd("hsetnx").arg(&keys.data).arg("dependencies").arg(serialize!(&cast.dependencies))
        .cmd("hsetnx").arg(&keys.data).arg("display_type").arg(serialize!(&cast.display_type))
        .cmd("hsetnx").arg(&keys.data).arg("output_collection").arg(serialize!(&cast.output_collection))
        .cmd("hsetnx").arg(&keys.data).arg("child_filters").arg(serialize!(&cast.child_filters))
        .cmd("hsetnx").arg(&keys.data).arg("network_policies").arg(serialize!(&cast.network_policies))
        .cmd("sadd").arg(&keys.set).arg(&cast.name);
    // add optional values if set
    hsetnx_opt_serialize!(pipe, &keys.data, "version", &cast.version);
    hsetnx_opt_serialize!(pipe, &keys.data, "image", &cast.image);
    hsetnx_opt_serialize!(pipe, &keys.data, "lifetime", &cast.lifetime);
    hsetnx_opt_serialize!(pipe, &keys.data, "timeout", &cast.timeout);
    hsetnx_opt_serialize!(pipe, &keys.data, "modifiers", &cast.modifiers);
    hsetnx_opt_serialize!(pipe, &keys.data, "description", &cast.description);
    hsetnx_opt_serialize!(pipe, &keys.data, "clean_up", &cast.clean_up);
    hsetnx_opt_serialize!(pipe, &keys.data, "kvm", &cast.kvm);
    // invalidate this images scaler cache
    pipe.cmd("hset").arg(&syskey.data).arg(cast.scaler.cache_key()).arg(true);
    Ok(())
}

/// Creates a group in the redis backend
///
/// # Arguments
///
/// * `user` - The user creating this group
/// * `request` - The image request to create in the backend
/// * `shared` - Shared objects in Thorium
#[rustfmt::skip]
pub async fn create(user: &User, request: ImageRequest, shared: &Shared) -> Result<Image, ApiError> {
    let settings = super::system::get_settings(shared).await?;
    // try to cast to an image
    let cast = request.cast(user, &settings)?;
    // build redis pipeline for saving this image
    let mut pipe = redis::pipe();
    build(&mut pipe, &cast, shared)?;
    // save image to backend
    let status: Vec<bool> = pipe.query_async(conn!(shared)).await?;
    // check if any errors occurred in all commands except for the final one as
    // hset will return 0 if the key already exists regardless of if we updated the value
    if status[..status.len() - 1].iter().any(|x|!x) {
        conflict!(
            format!("Image {} already exists in group {}", &cast.name, &cast.group)
        )
    } else {
        Ok(cast)
    }
}

/// Gets an images data from the backend
///
/// # Arguments
///
/// * `group` - The group to get an image From
/// * `name` - The name of the image to get
/// * `shared` - Shared objects in Thorium
pub async fn get(group: &str, name: &str, shared: &Shared) -> Result<Image, ApiError> {
    // build image keys
    let data_key = ImageKeys::data(group, name, shared);
    let used_by_key = ImageKeys::used_by(group, name, shared);
    // get image data and the names of pipelines using this image
    let (raw, used_by): (HashMap<String, String>, Vec<String>) = redis::pipe()
        .cmd("hgetall")
        .arg(data_key)
        .cmd("smembers")
        .arg(used_by_key)
        .query_async(conn!(shared))
        .await?;
    // check if an image was retrieved
    if raw.is_empty() {
        not_found!(format!("Image {}:{} not found", &group, name))
    } else {
        Image::try_from((raw, used_by))
    }
}

/// Gets info on list of images needed to make jobs based on this image
///
/// # Arguments
///
/// * `group` - The name of the group to get image info from
/// * `names` - The names of the images to get info about
/// * `shared` - Shared objects in Thorium
#[rustfmt::skip]
pub async fn job_info<'a>(
    group: &'a str,
    names: &'a [String],
    shared: &'a Shared,
) -> Result<HashMap<&'a String, ImageJobInfo>, ApiError> {
    // build a pipeline to get the image info needed for a job for all image names
    let mut pipe = redis::pipe();
    for name in names {
        // build key to this images data
        let key = ImageKeys::data(group, name, shared);
        // add command to get if this is a generator and what scaler it uses
        pipe.cmd("hget").arg(&key).arg("generator")
            .cmd("hget").arg(&key).arg("scaler");
    }
    // execute built to get a list of raw data
    let raw: Vec<(String, String)> = pipe.query_async(conn!(shared)).await?;
    // build a map of the return values
    let mut map = HashMap::with_capacity(names.len());
    for (i, item) in raw.iter().enumerate() {
        // build the image info for this image
        let info = ImageJobInfo {
            generator: coerce_bool!(&item.0, "generator"),
            scaler: deserialize!(&item.1, "scaler"),
        };
        map.insert(&names[i], info);
    }
    Ok(map)
}

/// Updates an image in the backend
///
/// # Arguments
///
/// * `image` - The image to update in the backend
/// * `shared` - Shared objects in Thorium
#[rustfmt::skip]
pub async fn update(image: &Image, shared: &Shared) -> Result<(), ApiError> {
    // build image keys
    let keys = ImageKeys::new(image, shared);
    // build key to system info
    let syskey = SystemKeys::new(shared);
    // build the pipeline to save this image with
    let mut pipe = redis::pipe();
    pipe.cmd("hset").arg(&keys.data).arg("scaler").arg(serialize!(&image.scaler))
        .cmd("hset").arg(&keys.data).arg("resources").arg(serialize!(&image.resources))
        .cmd("hset").arg(&keys.data).arg("spawn_limit").arg(serialize!(&image.spawn_limit))
        .cmd("hset").arg(&keys.data).arg("volumes").arg(serialize!(&image.volumes))
        .cmd("hset").arg(&keys.data).arg("env").arg(serialize!(&image.env))
        .cmd("hset").arg(&keys.data).arg("args").arg(serialize!(&image.args))
        .cmd("hset").arg(&keys.data).arg("security_context")
            .arg(serialize!(&image.security_context))
        .cmd("hset").arg(&keys.data).arg("collect_logs")
            .arg(serialize!(&image.collect_logs))
        .cmd("hset").arg(&keys.data).arg("generator").arg(serialize!(&image.generator))
        .cmd("hset").arg(&syskey.data).arg("scaler_cache").arg(true)
        .cmd("hset").arg(&keys.data).arg("dependencies").arg(serialize!(&image.dependencies))
        .cmd("hset").arg(&keys.data).arg("display_type").arg(serialize!(&image.display_type))
        .cmd("hset").arg(&keys.data).arg("output_collection").arg(serialize!(&image.output_collection))
        .cmd("hset").arg(&keys.data).arg("child_filters").arg(serialize!(&image.child_filters))
        .cmd("hset").arg(&keys.data).arg("bans").arg(serialize!(&image.bans))
        .cmd("hset").arg(&keys.data).arg("network_policies").arg(serialize!(&image.network_policies));
    // add optional values if set
    hset_del_opt_serialize!(pipe, &keys.data, "version", &image.version);
    hset_del_opt_serialize!(pipe, &keys.data, "image", &image.image);
    hset_del_opt_serialize!(pipe, &keys.data, "lifetime", &image.lifetime);
    hset_del_opt_serialize!(pipe, &keys.data, "timeout", &image.timeout);
    hset_del_opt_serialize!(pipe, &keys.data, "modifiers", &image.modifiers);
    hset_del_opt_serialize!(pipe, &keys.data, "description", &image.description);
    hset_del_opt_serialize!(pipe, &keys.data, "clean_up", &image.clean_up);
    hset_del_opt_serialize!(pipe, &keys.data, "kvm", &image.kvm);
    // invalidate this images scaler cache
    pipe.cmd("hset").arg(&syskey.data).arg(image.scaler.cache_key()).arg(true);
    // save image to backend
() = pipe.atomic().query_async(conn!(shared)).await?;
    Ok(())
}

/// Checks if an image exists in the Redis backend after authentication
///
/// Requiring a reference to a Group object obtained after authorization
/// decreases the likelihood of prematurely checking for the existence of
/// the image and leaking information to an unauthorized user
///
/// # Arguments
///
/// * `name` - The name of the image to check the existence of
/// * `group` - The image's group
/// * `shared` - Shared Thorium objects
pub async fn exists_authenticated<T: AsRef<str>>(
    name: T,
    group: &Group,
    shared: &Shared,
) -> Result<bool, ApiError> {
    let set_key = ImageKeys::set(&group.name, shared);
    helpers::exists(name.as_ref(), &set_key, shared).await
}

/// Gets the scaler an image is under
///
/// # Arguments
///
/// * `group` - The group this image is in
/// * `name` - The name of the image to inspect
/// * `shared` - Shared objects in Thorium
pub async fn get_scaler(group: &str, name: &str, shared: &Shared) -> Result<ImageScaler, ApiError> {
    // get group image set key
    let data_key = ImageKeys::data(group, name, shared);
    // query redis
    let raw: Option<String> = query!(cmd("hget").arg(data_key).arg("scaler"), shared).await?;
    // deserialize our scaler if we got a result
    if let Some(raw) = raw {
        // deserialize our scaler
        Ok(deserialize!(&raw))
    } else {
        not_found!(format!("Image {} does not exist in {}", name, group))
    }
}

/// Gets the scalers for multiple images
///
/// # Arguments
///
/// * `group` - The group this image is in
/// * `names` - The names of the images to inspect
/// * `shared` - Shared objects in Thorium
pub async fn get_scalers(
    group: &str,
    names: &[&String],
    shared: &Shared,
) -> Result<Vec<ImageScaler>, ApiError> {
    // create our redis pipeline
    let mut pipe = redis::pipe();
    // crawl over the names of the images to get scalers for
    for name in names {
        // build this images data key
        let data_key = ImageKeys::data(group, name, shared);
        // add the command to get this images scaler
        pipe.cmd("hget").arg(data_key).arg("scaler");
    }
    // query redis
    let scalers_strs: Vec<String> = pipe.atomic().query_async(conn!(shared)).await?;
    // build a vec to store our converted scalers
    let mut scalers = Vec::with_capacity(scalers_strs.len());
    // convert our strings to scalers
    for raw in scalers_strs {
        // deserialize our scaler
        scalers.push(deserialize!(&raw));
    }
    // if we find no scalers then return an error
    if scalers.len() == names.len() {
        Ok(scalers)
    } else {
        // we did not the correct number of scalers
        not_found!(format!(
            "All or some images do not have scalers: {}:{:?}",
            group, names
        ))
    }
}

/// Retrieve bans for an image from redis
///
/// If an image has no bans, the map will be empty
///
/// * `group` - The group this image is in
/// * `name` - The name of the images to inspect
/// * `shared` - Shared objects in Thorium
pub async fn get_bans(
    group: &str,
    name: &str,
    shared: &Shared,
) -> Result<HashMap<Uuid, ImageBan>, ApiError> {
    // create our data key
    let data_key = ImageKeys::data(group, name, shared);
    // query redis
    let raw: Option<String> = query!(cmd("hget").arg(data_key).arg("bans"), shared).await?;
    // cast to a ban map
    let bans = match raw {
        Some(raw) => deserialize!(&raw),
        None => None,
    };
    // return the bans or an empty HashMap if we got None
    Ok(bans.unwrap_or_default())
}

/// Lists all images in a group
///
/// # Arguments
///
/// * `group` - The group to list images from
/// * `cursor` - The cursor to use when paging through images
/// * `limit` - The number of objects to try and return (weakly enforced)
/// * `shared` - Shared Thorium objects
pub async fn list(
    group: &str,
    cursor: usize,
    limit: usize,
    shared: &Shared,
) -> Result<ImageList, ApiError> {
    // get group image set key
    let key = ImageKeys::set(group, shared);
    // get list of created groups
    let (new_cursor, names) = query!(
        cmd("sscan").arg(key).arg(cursor).arg("COUNT").arg(limit),
        shared
    )
    .await?;
    // cast to group list with correct cursor
    // if cursor is 0 no more groups exist
    if new_cursor == 0 {
        Ok(ImageList::new(None, names))
    } else {
        // more groups exist use new_cursor
        Ok(ImageList::new(Some(new_cursor), names))
    }
}

// The raw data from redis needed to build an image
type RawUserData = (HashMap<String, String>, Vec<String>);

/// Lists all images in a group with details
///
/// # Arguments
///
/// * `group` - The group to list images from
/// * `names` - The image names to get details on
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
#[instrument(name = "db::images::list_details", skip(shared), err(Debug))]
pub async fn list_details(
    group: &str,
    names: &[String],
    shared: &Shared,
) -> Result<Vec<Image>, ApiError> {
    // list image data
    let raw: Vec<RawUserData> = names.iter()
        .fold(redis::pipe().atomic(), |pipe, name|
            pipe.cmd("hgetall").arg(&ImageKeys::data(group, name, shared))
                .cmd("smembers").arg(&ImageKeys::used_by(group, name, shared))
            )
        .query_async(conn!(shared)).await?;
    // cast to image structs
    let images = cast!(raw, Image::try_from);
    Ok(images)
}

/// List all images in all groups with details for backups
///
/// # Arguments
///
/// * `shared` - Shared Thorium objects
#[instrument(name = "db::images::backup", skip_all, err(Debug))]
pub async fn backup(shared: &Shared) -> Result<Vec<Image>, ApiError> {
    // build key to group set
    let key = GroupKeys::set(shared);
    // get all group names
    let groups: Vec<String> = query!(cmd("smembers").arg(key), shared).await?;
    // build pipeline to retrieve all image data
    let mut pipe = redis::pipe();
    for group in &groups {
        // get all image names
        let image_key = ImageKeys::set(group, shared);
        let names: Vec<String> = query!(cmd("smembers").arg(image_key), shared).await?;
        names.iter().fold(&mut pipe, |pipe, name| {
            pipe.cmd("hgetall")
                .arg(ImageKeys::data(group, name, shared))
                .cmd("smembers")
                .arg(ImageKeys::used_by(group, name, shared))
        });
    }
    // execute pipeline to get all image data
    let raw: Vec<RawUserData> = pipe.query_async(conn!(shared)).await?;
    // cast to a vector of images
    let images = cast!(raw, Image::try_from);
    Ok(images)
}

/// Count the number of updates that will return true
///
/// # Arguments
///
/// * `images` - The images to check for updates for
fn update_counts(images: &[Image]) -> usize {
    // first count the non-optional fields (ones that should always be true)
    // this code is pretty ugly since it works off a magic number but there's
    // not really a better way ¯\_(ツ)_/¯
    let mut cnt = images.len() * 19;
    // count optional fields that contain a value for each image
    images.iter().for_each(|image| cnt += add_opts(image));
    cnt
}

/// Counts the number of optional fields in an image that contain a value
///
/// # Arguments
///
/// `image` - The image to count optionals from
fn add_opts(image: &Image) -> usize {
    let mut cnt: usize = 0;
    cnt += usize::from(image.version.is_some());
    cnt += usize::from(image.image.is_some());
    cnt += usize::from(image.lifetime.is_some());
    cnt += usize::from(image.timeout.is_some());
    cnt += usize::from(image.modifiers.is_some());
    cnt += usize::from(image.description.is_some());
    cnt += usize::from(image.clean_up.is_some());
    cnt += usize::from(image.kvm.is_some());
    cnt
}

/// Restore image data
///
/// # Arguments
///
/// * `users` - The list of images to restore
/// * `shared` - Shared Thorium objects
pub async fn restore(images: &[Image], shared: &Shared) -> Result<(), ApiError> {
    // build our redis pipeline
    let mut pipe = redis::pipe();
    // crawl over images and build the pipeline to restore each one
    images
        .iter()
        .map(|image| build(&mut pipe, image, shared))
        .collect::<Result<Vec<()>, ApiError>>()?;
    // try to save images into redis
    let status: Vec<bool> = pipe.atomic().query_async(conn!(shared)).await?;
    // get the number of commands that succeeded
    let real = status.into_iter().filter(|x| *x).count();
    let expected = update_counts(images);
    // check if restoring these images failed
    if real != expected {
        // failed to create image it must already exist
        return conflict!(format!(
            "Failed to restore images expected {expected} but found {real}"
        ));
    }
    Ok(())
}

/// Deletes an image from Redis
///
/// # Arguments
///
/// * `image` - The image to delete from redis
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn delete(image: &Image, shared: &Shared) -> Result<(), ApiError> {
    // build keys to image
    let keys = ImageKeys::new(image, shared);
    // build key to system info
    let syskey = SystemKeys::new(shared);
    // delete image from backend
    () = redis::pipe()
        .atomic()
        .cmd("del").arg(keys.data)
        .cmd("srem").arg(&keys.set).arg(&image.name)
        .cmd("hset").arg(&syskey.data).arg("scaler_cache").arg(true)
        .query_async(conn!(shared)).await?;
    // skip checking for success due to the fact that hset always returns false
    // delete all of the image's related data
    // delete all of the image's notifications
    image.delete_all_notifications(&ImageKey::from(image), shared).await?;
    // delete the used_by info for this image from all of its network policies
    super::network_policies::set_used_by(
        &image.group,
        std::iter::empty::<&str>(),
        image.network_policies.iter(),
        &image.name,
        shared
    ).await?;
    Ok(())
}

/// Deletes all images in a group
///
/// # Arguments
///
/// * `group` - The group to delete all images from
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn delete_all(group: &Group, shared: &Shared) -> Result<(), ApiError> {
    // loop until jobs in this stage are deleted
    let mut cursor = 0;
    loop {
        // get a list of images in the group
        let images = Image::list(group, cursor, 1000, shared).await?;
        // delete image data
        images.names.iter()
            .fold(redis::pipe().atomic(), |pipe, name|
                pipe.cmd("del").arg(ImageKeys::data(&group.name, name, shared))
                    .cmd("del").arg(ImageKeys::used_by(&group.name, name, shared)))
            .exec_async(conn!(shared)).await?;
        // delete all of the objects owned by the images
        let mut images = images.details(group, shared).await?;
        for image in images.details.drain(..) {
            let image_key = ImageKey::new(&group.name, &image.name);
            // delete image data concurrently
            tokio::try_join!(
                // delete all of the image's notifications
                super::notifications::delete_all::<Image>(&image_key, shared),
                // remove the image from the "used_by" list for all of its network policies
                super::network_policies::set_used_by(
                    &group.name,
                    std::iter::empty::<&str>(),
                    image.network_policies.iter(),
                    &image.name,
                    shared
                )
            )?;
        }

        // check if our cursor has been exhausted
        if images.cursor.is_none() {
            break;
        }
        // update cursor
        cursor = images.cursor.unwrap();
    }
    // delete the images set for this group
    exec_query!(cmd("del").arg(ImageKeys::set(&group.name, shared)), shared).await?;
    Ok(())
}

/// Updates all images average runtimes in a group
///
/// # Arguments
///
/// * `group` - The group to update the average runtime for within a group
/// * `shared` - Shared Thorium objects
#[rustfmt::skip]
pub async fn update_runtimes(group: &Group, shared: &Shared) -> Result<(), ApiError> {
    // loop until jobs in this stage are deleted
    let mut cursor = 0;
    loop {
        let images = Image::list(group, cursor, 1000, shared).await?;
        // iterate over images and calculate and update their average runtime
        for image in images.names {
            // calculate and update the average runtime over the last 10k runs
            // for this image
            let script = redis::Script::new(
                r"
            local times = redis.call('lrange', ARGV[1], '0', 10000);
            local sum = 0;
            local total = 0;
            if #times ~= 0 then
                for i=1, #times, 1 do
                    sum = sum + times[i];
                    total = total + 1;
                end
                local avg = sum / total;
                redis.call('hset', ARGV[2], 'runtime', avg);
            end"
            );
            let _: bool = script
                .arg(ImageKeys::ttc_queue(&group.name, &image, shared))
                .arg(ImageKeys::data(&group.name, &image, shared))
                .invoke_async(conn!(shared))
                .await?;
        }
        // check if our cursor has been exhausted
        if images.cursor.is_none() {
            break;
        }
        // update cursor
        cursor = images.cursor.unwrap();
    }
    Ok(())
}
