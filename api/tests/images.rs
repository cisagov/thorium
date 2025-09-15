//! Tests the Images routes in Thorium

use std::collections::HashSet;
use std::path::PathBuf;

use futures::{stream, StreamExt, TryStreamExt};
use thorium::models::{
    ArgStrategy, AutoTagLogic, AutoTagUpdate, ChildFilters, ChildFiltersUpdate, CleanupUpdate,
    DependenciesUpdate, DependencyPassStrategy, DependencySettingsUpdate,
    EphemeralDependencySettingsUpdate, FilesHandlerUpdate, GroupUpdate, GroupUsersUpdate,
    HostPathWhitelistUpdate, ImageBan, ImageBanKind, ImageBanUpdate, ImageLifetime,
    ImageNetworkPolicyUpdate, ImageScaler, ImageUpdate, ImageVersion, NetworkPolicyRequest,
    NotificationLevel, NotificationParams, NotificationRequest, OutputCollectionUpdate,
    OutputDisplayType, OutputHandler, PipelineRequest, ResourcesUpdate,
    ResultDependencySettingsUpdate, SystemSettingsResetParams, SystemSettingsUpdate,
    SystemSettingsUpdateParams, Volume, VolumeTypes,
};
use thorium::test_utilities::{self, generators};
use thorium::{contains, fail, is, is_in, unwrap_variant, vec_in_vec, Error};
use uuid::Uuid;

#[tokio::test]
async fn create() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // Create a test image
    let image_req = generators::gen_image(&group);
    let resp = client.images.create(&image_req).await?;
    is!(resp.status().as_u16(), 204);
    Ok(())
}

#[tokio::test]
async fn create_bad_volume_name() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // attempt to create an image with an invalid name
    let image_req = generators::gen_image(&group).volume(Volume::new(
        "InvalidImageName!@#$",
        "/placeholder",
        VolumeTypes::Secret,
    ));
    let resp = client.images.create(&image_req).await;
    // expect a BAD error
    fail!(resp, 400, "volume name must be only lowercase alphanumeric");
    // attempt to create an image with a reserved thorium name
    let image_req = generators::gen_image(&group).volume(Volume::new(
        "thorium-thingy",
        "/placeholder",
        VolumeTypes::Secret,
    ));
    let resp = client.images.create(&image_req).await;
    // expect a BAD error
    fail!(resp, 400, "Volume names cannot start with 'thorium'");
    Ok(())
}

#[tokio::test]
async fn create_conflict() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a test image
    let mut image_req = generators::gen_image(&group);
    // set the description as blank
    image_req.description = None;
    let resp = client.images.create(&image_req).await?;
    is!(resp.status().as_u16(), 204);
    // attempt to save the image again, this time with the optional description set
    image_req = image_req.description("This description should not be set for the existing image");
    let resp = client.images.create(&image_req).await;
    // expect a conflict error
    fail!(resp, 409);
    // check that the description was NOT set
    let image = client.images.get(&group, &image_req.name).await?;
    is!(image.description, None::<String>);
    Ok(())
}

#[tokio::test]
async fn create_bad_child_filter() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // attempt to create an image with a bad child filter regular expression
    let image_req = generators::gen_image(&group)
        .child_filters(ChildFilters::default().mime(r"incomplete-escape\"));
    let resp = client.images.create(&image_req).await;
    fail!(resp, 400, "filter regular expressions is invalid");
    Ok(())
}

#[serial_test::serial]
#[tokio::test]
async fn create_host_path() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // reset settings with no scan
    client
        .system
        .reset_settings(&SystemSettingsResetParams::default().no_scan())
        .await?;
    // get user client
    let user_client = generators::client(&client).await?;
    // create a group
    let group = generators::groups(1, &user_client).await?.remove(0).name;
    // attempt to create an image with an un-whitelisted host path
    let image_req = generators::gen_host_path(&group, "/disallowed/mount");
    let resp = user_client.images.create(&image_req).await;
    // expect a 400 error
    fail!(resp, 400, "not in the list of allowed host paths");
    // create that same image as an admin and expect the same result
    let resp = client.images.create(&image_req).await;
    fail!(resp, 400, "not in the list of allowed host paths");
    // update the whitelist with a path
    let whitelisted_path = "/whitelisted/path";
    client
        .system
        .update_settings(
            &SystemSettingsUpdate::default()
                .host_path_whitelist(HostPathWhitelistUpdate::default().add_path(whitelisted_path)),
            &SystemSettingsUpdateParams::default().no_scan(),
        )
        .await?;
    // successfully create an image with the whitelisted host path as a user
    let image_req = generators::gen_host_path(&group, whitelisted_path);
    user_client.images.create(&image_req).await?;
    // successfully create an image with a host path whose parent is in the whitelist
    let image_req =
        generators::gen_host_path(&group, format!("{whitelisted_path}/child/grandchild"));
    user_client.images.create(&image_req).await?;
    // allow unrestricted host paths
    client
        .system
        .update_settings(
            &SystemSettingsUpdate::default().allow_unrestricted_host_paths(true),
            &SystemSettingsUpdateParams::default().no_scan(),
        )
        .await?;
    // successfully create an image with a host path not on the whitelist as a user
    let image_req = generators::gen_host_path(&group, "/not/whitelisted/but/unrestricted");
    user_client.images.create(&image_req).await?;
    // attempt to create a relative (and therefore invalid) host path
    let image_req = generators::gen_host_path(&group, "relative/so/invalid");
    let resp = user_client.images.create(&image_req).await;
    fail!(resp, 400, "Host paths must be absolute");
    // attempt to create an invalid host path with relative traversal
    let image_req = generators::gen_host_path(&group, "/absolute/but/has/...../so/bad");
    let resp = user_client.images.create(&image_req).await;
    fail!(resp, 400, "must not contain relative traversal");
    // disallow unrestricted host paths again
    client
        .system
        .update_settings(
            &SystemSettingsUpdate::default().allow_unrestricted_host_paths(false),
            &SystemSettingsUpdateParams::default().no_scan(),
        )
        .await?;
    // make sure we can't mount to un-whitelisted paths after we restrict host paths again
    let image_req = generators::gen_host_path(&group, "/not/whitelisted/but/unrestricted");
    let resp = user_client.images.create(&image_req).await;
    fail!(resp, 400, "not in the list of allowed host paths");
    // make sure we can't mount to un-whitelisted paths after we restrict host paths again
    let image_req = generators::gen_host_path(&group, "/not/whitelisted/but/unrestricted");
    let resp = user_client.images.create(&image_req).await;
    fail!(resp, 400, "not in the list of allowed host paths");
    // reset settings
    client
        .system
        .reset_settings(&SystemSettingsResetParams::default().no_scan())
        .await?;
    Ok(())
}

#[tokio::test]
async fn create_network_policy() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create groups
    let groups = generators::groups(2, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    // create network policies in those groups
    let network_policies = generators::network_policies(&groups, 2, &client).await?;
    // create an image with those network policies that's in one of the policies' groups
    let mut image_req = generators::gen_image(&groups[0]);
    image_req = image_req.network_policies(network_policies.into_iter().map(|p| p.name));
    let resp = client.images.create(&image_req).await?;
    is!(resp.status().as_u16(), 204);
    Ok(())
}

#[tokio::test]
async fn create_bad_network_policy() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let image_group = generators::groups(1, &client).await?.remove(0).name;
    // create an image with a network policy that doesn't exist
    let mut image_req = generators::gen_image(&image_group);
    image_req = image_req.network_policy("network-policy-no-exist");
    let resp = client.images.create(&image_req).await;
    fail!(resp, 404);
    // create an image with a network policy that's not in the image's group
    let policy_group = generators::groups(1, &client).await?.remove(0).name;
    let network_policy = generators::network_policies(&[policy_group.clone()], 1, &client)
        .await?
        .remove(0);
    image_req = image_req.network_policy(&network_policy.name);
    let resp = client.images.create(&image_req).await;
    fail!(resp, 404);
    // try to create a new image not scaled by K8's but with a network policy
    let image_req = generators::gen_ext_image(&policy_group).network_policy(&network_policy.name);
    let resp = client.images.create(&image_req).await;
    fail!(resp, 400, "only be applied to images scaled in K8s");
    Ok(())
}

#[tokio::test]
async fn create_default_network_policies() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create group
    let group = generators::groups(1, &client).await?.remove(0).name;
    let group_array = [group.clone()];
    // create default network policies in that group
    let default_policy_reqs: Vec<NetworkPolicyRequest> = (0..10)
        .map(|_| {
            let mut req = generators::gen_network_policy(&group_array);
            req.default_policy = true;
            req
        })
        .collect();
    stream::iter(default_policy_reqs.iter().cloned())
        .map(Ok::<NetworkPolicyRequest, thorium::Error>)
        .try_for_each_concurrent(50, |req| {
            let client_ref = &client;
            async move {
                client_ref.network_policies.create(req).await?;
                Ok(())
            }
        })
        .await?;
    // create an image with no network policies
    let mut image_req = generators::gen_image(&group);
    image_req.network_policies = HashSet::new();
    client.images.create(&image_req).await?;
    // make sure that image has all default network policies
    let image = client.images.get(&group, &image_req.name).await?;
    for policy_name in default_policy_reqs.iter().map(|policy| &policy.name) {
        contains!(image.network_policies, policy_name);
    }
    Ok(())
}

#[tokio::test]
async fn get() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create an image and then get it
    let image = generators::gen_image(&group);
    let resp = client.images.create(&image).await?;
    is!(resp.status().as_u16(), 204);
    // get the image and compare it
    let retrieved = client.images.get(&group, &image.name).await?;
    is!(retrieved, image);
    Ok(())
}

#[tokio::test]
async fn list() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 20 random images
    let images = generators::images(&group, 20, false, &client).await?;
    // get the names of all the images we have created
    let names: Vec<String> = images.iter().map(|images| images.name.clone()).collect();
    // list the images we just created
    let mut cursor = client.images.list(&group);
    cursor.next().await?;
    // make sure all the images we tried to create are in our list
    for image in names {
        is_in!(cursor.names, image);
    }
    Ok(())
}

#[tokio::test]
async fn list_details() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 20 random images
    let images = generators::images(&group, 20, false, &client).await?;
    // list theimages we just created
    let mut cursor = client.images.list(&group).details();
    cursor.next().await?;
    // make sure all the group details we tried to create are in our list
    vec_in_vec!(&cursor.details, &images);
    Ok(())
}

#[tokio::test]
async fn delete() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup a random image
    let image = generators::images(&group, 1, false, &client)
        .await?
        .remove(0);
    // delete that image
    client.images.delete(&group, &image.name).await?;
    // make sure the image was deleted
    let resp = client.images.get(&group, &image.name).await;
    fail!(resp, 404);
    // TODO: test that all notifications were deleted; can't do that without specific
    // route to get notifications because we get a 404 when we try to get notifications
    // after deletion
    Ok(())
}

#[tokio::test]
async fn delete_conflict() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup a random image
    let image = generators::images(&group, 1, false, &client)
        .await?
        .remove(0);
    // build our pipeline request
    let order = serde_json::json!(vec![&image.name]);
    let pipe_req = PipelineRequest::new(&group, "testpipe", order);
    // create a pipeline that uses this image
    client.pipelines.create(&pipe_req).await?;
    // delete that image
    let status = client.images.delete(&group, &image.name).await;
    fail!(status, 409);
    Ok(())
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn update() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create network policies
    let network_policies = generators::network_policies(&[group.clone()], 2, &client).await?;
    let default_policy = generators::gen_network_policy(&[group.clone()]).default_policy();
    client
        .network_policies
        .create(default_policy.clone())
        .await?;
    // setup a random image
    let image = generators::images(&group, 1, false, &client)
        .await?
        .remove(0);
    // build and update for that image
    let update = ImageUpdate::default()
        .version(ImageVersion::SemVer(semver::Version::parse("1.1.0-RC")?))
        .image("rust:1.48.0")
        .lifetime(ImageLifetime::jobs(12))
        .timeout(123_452)
        .resources(
            ResourcesUpdate::default()
                .millicpu(2600)
                .memory("4Gi")
                .storage("128Gi")
                .nvidia_gpu(3)
                .amd_gpu(1),
        )
        .add_volume(Volume::new("test-vol", "/files", VolumeTypes::ConfigMap))
        .description("edited description")
        .remove_volume("woot")
        .disable_logs()
        .enable_generator()
        .dependencies(
            DependenciesUpdate::default()
                .samples(
                    DependencySettingsUpdate::default()
                        .location("/updated/path")
                        .kwarg("--update")
                        .strategy(DependencyPassStrategy::Names),
                )
                .ephemeral(
                    EphemeralDependencySettingsUpdate::default()
                        .location("/updated/ephemeral/path")
                        .kwarg("--ephemeral")
                        .strategy(DependencyPassStrategy::Names)
                        .add_name("updated.txt")
                        .remove_name("file.txt"),
                )
                .results(
                    ResultDependencySettingsUpdate::default()
                        .image("new-harvest")
                        .remove_image("harvest")
                        .location("/new/location")
                        .kwarg(thorium::models::KwargDependency::List("--new".to_owned()))
                        .name("new-fields.txt")
                        .remove_name("field.txt"),
                )
                .repos(
                    DependencySettingsUpdate::default()
                        .location("/new/location")
                        .kwarg("--new-repos")
                        .strategy(DependencyPassStrategy::Disabled),
                ),
        )
        .display_type(OutputDisplayType::Json)
        .output_collection(
            OutputCollectionUpdate::default()
                .handler(OutputHandler::Files)
                .files(
                    FilesHandlerUpdate::default()
                        .results("/updated/results")
                        .result_files("/updated/result_files")
                        .tags("/updated/tags")
                        .add_name("corn.csv")
                        .remove_name("corn.json"),
                )
                .auto_tag(
                    "Plant",
                    AutoTagUpdate::default().logic(AutoTagLogic::Equal(serde_json::json!("Corn"))),
                ),
        )
        .child_filters(
            ChildFiltersUpdate::default()
                .add_file_extensions(["exe", "txt", "so"])
                .submit_non_matches(true),
        )
        .clean_up(
            CleanupUpdate::default()
                .script("/updated/script.py".to_owned())
                .job_id(ArgStrategy::Kwarg("--new_job_id".to_owned()))
                .results(ArgStrategy::Append)
                .result_files_dir(ArgStrategy::Kwarg("--output_dir".to_owned())),
        )
        .bans(
            ImageBanUpdate::default()
                .add_ban(ImageBan::new(ImageBanKind::generic("Test ban 1!")))
                .add_ban(ImageBan::new(ImageBanKind::generic("Test ban 2!"))),
        )
        .network_policies(
            ImageNetworkPolicyUpdate::default()
                // add policies
                .add_policies(network_policies.into_iter().map(|p| p.name))
                // remove default policy
                .remove_policy(default_policy.name),
        );
    // update that image and check the response code
    client.images.update(&group, &image.name, &update).await?;
    // get the image and make sure our updates were applied
    let updated = client.images.get(&group, &image.name).await?;
    is!(updated, update);
    // make sure that the scaler cache is set to be cleared
    let info = client.system.get_info(Some(ImageScaler::K8s)).await?;
    is!(info.expired_cache(ImageScaler::K8s), true);
    Ok(())
}

#[tokio::test]
async fn update_bad_name() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup a random image
    let image = generators::images(&group, 1, false, &client)
        .await?
        .remove(0);
    // build and update for that image
    let update = ImageUpdate::default()
        .image("rust:1.48.0")
        .lifetime(ImageLifetime::jobs(12))
        .timeout(123_452)
        .resources(
            ResourcesUpdate::default()
                .millicpu(2600)
                .memory("4Gi")
                .storage("128Gi")
                .nvidia_gpu(3)
                .amd_gpu(1),
        )
        .add_volume(Volume::new("Test_vol**", "/files", VolumeTypes::ConfigMap))
        .remove_volume("woot");
    // update that image and check the response code
    let resp = client.images.update(&group, &image.name, &update).await;
    fail!(resp, 400);
    Ok(())
}

#[tokio::test]
async fn update_user() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // get a user client
    let user_client = generators::client(&client).await?;
    // Create a group
    let group = generators::groups(1, &user_client).await?.remove(0).name;
    // setup a random image
    let image = generators::images(&group, 1, false, &user_client)
        .await?
        .remove(0);
    // build and update for that image
    let update = ImageUpdate::default()
        .image("rust:1.48.0")
        .lifetime(ImageLifetime::jobs(12))
        .timeout(3)
        .resources(
            ResourcesUpdate::default()
                .millicpu(2600)
                .memory("4Gi")
                .storage("128Gi")
                .nvidia_gpu(3)
                .amd_gpu(1),
        )
        .add_volume(Volume::new("test-vol", "/files", VolumeTypes::ConfigMap))
        .remove_volume("woot");
    // update that image and check the response code
    user_client
        .images
        .update(&group, &image.name, &update)
        .await?;
    // get the image and make sure our updates were applied
    let updated = user_client.images.get(&group, &image.name).await?;
    is!(updated, update);
    // make sure that the scaler cache is set to be cleared
    let info = client.system.get_info(Some(ImageScaler::K8s)).await?;
    is!(info.expired_cache(ImageScaler::K8s), true);
    Ok(())
}

#[tokio::test]
async fn update_clear_description() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup a random image
    let image = generators::images(&group, 1, false, &client)
        .await?
        .remove(0);
    // update the image with some description
    let update = ImageUpdate::default().description("edited description");
    client.images.update(&group, &image.name, &update).await?;
    let updated = client.images.get(&group, &image.name).await?;
    is!(updated.description, update.description);
    // now clear the description with a new ImageUpdate
    let update = ImageUpdate::default().clear_description();
    client.images.update(&group, &image.name, &update).await?;
    let updated = client.images.get(&group, &image.name).await?;
    // ensure that description is empty
    is!(updated.description, Option::<String>::None);
    Ok(())
}

#[tokio::test]
async fn update_clear_version() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup a random image
    let image = generators::images(&group, 1, false, &client)
        .await?
        .remove(0);
    // update the image with some version
    let update = ImageUpdate::default().version(ImageVersion::Custom("custom_v1".to_string()));
    client.images.update(&group, &image.name, &update).await?;
    let updated = client.images.get(&group, &image.name).await?;
    is!(updated, update);
    is!(updated.version, update.version);
    // now clear the version with a new ImageUpdate
    let update = ImageUpdate::default().clear_version();
    client.images.update(&group, &image.name, &update).await?;
    let updated = client.images.get(&group, &image.name).await?;
    // ensure that version is empty
    is!(updated, update);
    is!(updated.version, Option::<ImageVersion>::None);
    Ok(())
}

#[tokio::test]
async fn update_child_filters() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create the image with a child filter
    let child_filter = r"remove-me";
    let image_req =
        generators::gen_image(&group).child_filters(ChildFilters::default().mime(child_filter));
    client.images.create(&image_req).await?;
    // add a regular expression to it and remove the existing one
    let update = ImageUpdate::default().child_filters(
        ChildFiltersUpdate::default()
            .add_mime(r"new-filter")
            .remove_mime(child_filter),
    );
    client
        .images
        .update(&group, &image_req.name, &update)
        .await?;
    // get the updated image
    let image = client.images.get(&group, &image_req.name).await?;
    // make sure the update applied
    is!(image, update);
    Ok(())
}

#[tokio::test]
async fn update_bad_child_filters() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup a random image
    let image = generators::images(&group, 1, false, &client)
        .await?
        .remove(0);
    // try to add a bad regular expression to it
    let update = ImageUpdate::default()
        .child_filters(ChildFiltersUpdate::default().add_mime(r"unrecognized-escape\q"));
    let resp = client.images.update(&group, &image.name, &update).await;
    fail!(resp, 400, "filter regular expressions is invalid");
    // try to remove a regular expression it doesn't have
    let update = ImageUpdate::default()
        .child_filters(ChildFiltersUpdate::default().remove_mime(r"not-found"));
    let resp = client.images.update(&group, &image.name, &update).await;
    fail!(resp, 400, "missing one or more mime child filters");
    Ok(())
}

#[tokio::test]
async fn update_bans() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup a random image
    let image = generators::images(&group, 1, false, &client)
        .await?
        .remove(0);
    // update the image with bans
    let mut bans = vec![
        ImageBan::new(ImageBanKind::generic("Test ban 1!")),
        ImageBan::new(ImageBanKind::image_url(
            image.image.clone().unwrap_or_default(),
        )),
    ];
    let update = ImageUpdate::default().bans(ImageBanUpdate::default().add_bans(bans.clone()));
    client.images.update(&group, &image.name, &update).await?;
    // successfully remove a ban from the image
    let update = ImageUpdate::default().bans(
        ImageBanUpdate::default().remove_ban(
            bans.pop()
                .ok_or_else(|| Error::new("Popped empty bans vec"))?
                .id,
        ),
    );
    client.images.update(&group, &image.name, &update).await?;
    // get the image and make sure the ban was removed
    let updated = client.images.get(&group, &image.name).await?;
    is!(updated, update);
    // attempt to remove a ban as a non admin
    let user_client = generators::client(&client).await?;
    let username = user_client.users.info().await?.username;
    // add the user to the group
    let group_update =
        GroupUpdate::default().users(GroupUsersUpdate::default().direct_add(username));
    client.groups.update(&group, &group_update).await?;
    let update = ImageUpdate::default().bans(
        ImageBanUpdate::default().remove_ban(
            bans.pop()
                .ok_or_else(|| Error::new("Popped empty bans vec"))?
                .id,
        ),
    );
    let resp = user_client
        .images
        .update(&group, &image.name, &update)
        .await;
    fail!(resp, 401);
    // remove the second ban
    client.images.update(&group, &image.name, &update).await?;
    // get the image and make sure the ban was removed
    let updated = client.images.get(&group, &image.name).await?;
    is!(updated, update);
    // attempt to remove a ban that doesn't exist
    let update = ImageUpdate::default().bans(ImageBanUpdate::default().remove_ban(Uuid::new_v4()));
    let resp = client.images.update(&group, &image.name, &update).await;
    fail!(resp, 404);
    Ok(())
}

#[serial_test::serial]
#[tokio::test]
async fn update_fix_ban() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // get a user client
    let user_client = generators::client(&client).await?;
    // Create a group
    let group = generators::groups(1, &user_client).await?.remove(0).name;
    // reset settings
    client
        .system
        .reset_settings(&SystemSettingsResetParams::default().no_scan())
        .await?;
    // allow all host paths
    let settings_update = SystemSettingsUpdate::default().allow_unrestricted_host_paths(true);
    client
        .system
        .update_settings(
            &settings_update,
            &SystemSettingsUpdateParams::default().no_scan(),
        )
        .await?;
    // create an image with a host path
    let vol_path = "/some/path";
    let image_req = generators::gen_host_path(&group, vol_path);
    client.images.create(&image_req).await?;
    // get the host path volume
    let host_path_volume = image_req
        .volumes
        .iter()
        .find(|vol| match vol.archetype {
            VolumeTypes::HostPath => vol.host_path.is_some(),
            _ => false,
        })
        .unwrap();
    // set a host path ban on that image
    let image_update = ImageUpdate::default().bans(ImageBanUpdate::default().add_ban(
        ImageBan::new(ImageBanKind::host_path(&host_path_volume.name, vol_path)),
    ));
    client
        .images
        .update(&group, &image_req.name, &image_update)
        .await?;
    // verify that the ban is set
    let image = user_client.images.get(&group, &image_req.name).await?;
    is!(image.bans.len(), 1, "Set ban");
    let ban = image.bans.values().next().unwrap();
    let ban = unwrap_variant!(&ban.ban_kind, ImageBanKind::InvalidHostPath);
    is!(
        PathBuf::from(vol_path),
        ban.host_path,
        "Banned host path is correct"
    );
    // remove the problematic volume
    let image_update = ImageUpdate::default().remove_volume(&ban.volume_name);
    user_client
        .images
        .update(&group, &image.name, &image_update)
        .await?;
    // verify that the ban was removed
    let image = user_client.images.get(&group, &image_req.name).await?;
    is!(image.bans.len(), 0, "Ban removed after bad volume removed");
    // reset settings
    client
        .system
        .reset_settings(&SystemSettingsResetParams::default().no_scan())
        .await?;
    Ok(())
}

#[tokio::test]
async fn update_network_policy() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create groups
    let groups = generators::groups(2, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    // create network policies in those groups
    let network_policies = generators::network_policies(&groups, 2, &client).await?;
    // create an image with those network policies that's in one of the policies' groups
    let mut image_req = generators::gen_image(&groups[0]);
    image_req = image_req.network_policies(network_policies.iter().map(|p| &p.name));
    client.images.create(&image_req).await?;
    // remove one of the network policies
    let update = ImageUpdate::default().network_policies(
        ImageNetworkPolicyUpdate::default().remove_policy(&network_policies[0].name),
    );
    client
        .images
        .update(&image_req.group, &image_req.name, &update)
        .await?;
    let image = client.images.get(&image_req.group, &image_req.name).await?;
    is!(
        image.network_policies.len(),
        1,
        "network policies length is 1 after remove policy"
    );
    is!(
        image.network_policies.iter().next().unwrap(),
        &network_policies[1].name,
        "policy is policy 1 after add/remove same policy"
    );
    // remove the one network policy and add the other one
    let update = ImageUpdate::default().network_policies(
        ImageNetworkPolicyUpdate::default()
            .add_policy(&network_policies[0].name)
            .remove_policy(&network_policies[1].name),
    );
    client
        .images
        .update(&image_req.group, &image_req.name, &update)
        .await?;
    let image = client.images.get(&image_req.group, &image_req.name).await?;
    is!(
        image.network_policies.len(),
        1,
        "network policies length is 1 after add/remove policies"
    );
    is!(
        image.network_policies.iter().next().unwrap(),
        &network_policies[0].name,
        "policy is policy 0 after add/remove same policy"
    );
    Ok(())
}

#[tokio::test]
async fn update_bad_network_policy() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create groups
    let image_group = generators::groups(1, &client).await?.remove(0).name;
    let policy_group = generators::groups(1, &client).await?.remove(0).name;
    // create image with a policy
    let image_policy = generators::network_policies(&[image_group.clone()], 1, &client)
        .await?
        .remove(0);
    let mut image_req = generators::gen_image(&image_group);
    image_req = image_req.network_policy(&image_policy.name);
    client.images.create(&image_req).await?;
    // create network policy
    let not_image_policy = generators::network_policies(&[policy_group.clone()], 1, &client)
        .await?
        .remove(0);
    // attempt to add a network policy that is not in the image's group
    let update = ImageUpdate::default()
        .network_policies(ImageNetworkPolicyUpdate::default().add_policy(&not_image_policy.name));
    let resp = client
        .images
        .update(&image_req.group, &image_req.name, &update)
        .await;
    fail!(resp, 404);
    // attempt to remove a network policy that the image does not have
    let update = ImageUpdate::default()
        .network_policies(ImageNetworkPolicyUpdate::default().remove_policy("policy-no-exist"));
    let resp = client
        .images
        .update(&image_req.group, &image_req.name, &update)
        .await;
    fail!(resp, 400);
    // attempt to add a network policy the image already has
    let update = ImageUpdate::default()
        .network_policies(ImageNetworkPolicyUpdate::default().add_policy(&image_policy.name));
    let resp = client
        .images
        .update(&image_req.group, &image_req.name, &update)
        .await;
    fail!(resp, 400);
    // attempt to add a network policy that does not exist at all
    let update = ImageUpdate::default()
        .network_policies(ImageNetworkPolicyUpdate::default().add_policy("policy-no-exist"));
    let resp = client
        .images
        .update(&image_req.group, &image_req.name, &update)
        .await;
    fail!(resp, 404);
    // try to change the image's scaler type while it has a network policy
    let update = ImageUpdate::default().scaler(ImageScaler::External);
    let resp = client
        .images
        .update(&image_req.group, &image_req.name, &update)
        .await;
    fail!(resp, 400, "non-K8's while network policies are applied");
    // create a new image not scaled by K8's
    let image_req = generators::gen_ext_image(&image_group);
    client.images.create(&image_req).await?;
    // try to add a network policy to an image not scaled in K8's
    let update = ImageUpdate::default()
        .network_policies(ImageNetworkPolicyUpdate::default().add_policy(&image_policy.name));
    let resp = client
        .images
        .update(&image_req.group, &image_req.name, &update)
        .await;
    fail!(resp, 400, "only be applied to images scaled in K8s");
    Ok(())
}

#[tokio::test]
async fn notifications_bans() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // get user client
    let user_client = generators::client(&client).await?;
    // Create a group
    let group = generators::groups(1, &user_client).await?.remove(0).name;
    // setup a random image
    let image = generators::images(&group, 1, false, &user_client)
        .await?
        .remove(0);
    // update the image with bans
    let generic_ban_msg = "Test ban 1!";
    let mut bans = vec![
        ImageBan::new(ImageBanKind::generic(generic_ban_msg)),
        ImageBan::new(ImageBanKind::image_url(
            image.image.clone().unwrap_or_default(),
        )),
    ];
    let update = ImageUpdate::default().bans(ImageBanUpdate::default().add_bans(bans.clone()));
    client.images.update(&group, &image.name, &update).await?;
    // get the images notifications and make sure there was a notification added for the ban
    let notifications = user_client
        .images
        .get_notifications(&group, &image.name)
        .await?;
    is!(notifications.len(), 2);
    let mut ban_ids: Vec<Uuid> = bans.iter().map(|ban| ban.id).collect();
    contains!(ban_ids, notifications[0].ban_id.as_ref().unwrap());
    contains!(ban_ids, notifications[1].ban_id.as_ref().unwrap());
    // successfully remove a ban from the image
    let update = ImageUpdate::default().bans(
        ImageBanUpdate::default().remove_ban(
            bans.pop()
                .ok_or_else(|| Error::new("Popped empty bans vec"))?
                .id,
        ),
    );
    client.images.update(&group, &image.name, &update).await?;
    // get the images notifications and make sure that notification was automatically removed
    let notifications = user_client
        .images
        .get_notifications(&group, &image.name)
        .await?;
    is!(notifications.len(), 1);
    ban_ids.pop();
    contains!(ban_ids, notifications[0].ban_id.as_ref().unwrap());
    is!(notifications[0].msg, generic_ban_msg);
    Ok(())
}

#[tokio::test]
async fn create_notification() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup a random image
    let image = generators::images(&group, 1, false, &client)
        .await?
        .remove(0);
    // create an image notification
    let req = NotificationRequest::new("Test warning message!", NotificationLevel::Warn);
    client
        .images
        .create_notification(&group, &image.name, &req, &NotificationParams::default())
        .await?;
    // make sure the image notification is there
    let notifications = client.images.get_notifications(&group, &image.name).await?;
    is!(notifications.len(), 1);
    is!(notifications[0], req);
    is!(notifications[0].key.group, group);
    is!(notifications[0].key.image, image.name);
    Ok(())
}

#[tokio::test]
async fn create_notification_bad() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // get user client
    let user_client = generators::client(&client).await?;
    // Create a group
    let group = generators::groups(1, &user_client).await?.remove(0).name;
    // setup a random image
    let image = generators::images(&group, 1, false, &user_client)
        .await?
        .remove(0);
    // fail to create an image notification as a regular user
    let req = NotificationRequest::new("Test warning message!", NotificationLevel::Warn);
    let resp = user_client
        .images
        .create_notification(&group, &image.name, &req, &NotificationParams::default())
        .await;
    fail!(resp, 401);
    // fail to create an image notification for an image that doesn't exist
    let resp = client
        .images
        .create_notification(
            &group,
            "does-not-exist",
            &req,
            &NotificationParams::default(),
        )
        .await;
    fail!(resp, 404);
    Ok(())
}

#[tokio::test]
async fn delete_notification() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup a random image
    let image = generators::images(&group, 1, false, &client)
        .await?
        .remove(0);
    // create a notification
    let req = NotificationRequest::new("Test warning message!", NotificationLevel::Warn);
    client
        .images
        .create_notification(&group, &image.name, &req, &NotificationParams::default())
        .await?;
    // get the image notification
    let notification = client
        .images
        .get_notifications(&group, &image.name)
        .await?
        .remove(0);
    // delete the notification
    client
        .images
        .delete_notification(&group, &image.name, &notification.id)
        .await?;
    // check that the notification was deleted
    let notifications = client.images.get_notifications(&group, &image.name).await?;
    is!(notifications.len(), 0);
    Ok(())
}

#[tokio::test]
async fn delete_notification_bad() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // get user client
    let user_client = generators::client(&client).await?;
    // Create a group
    let group = generators::groups(1, &user_client).await?.remove(0).name;
    // setup a random image
    let image = generators::images(&group, 1, false, &user_client)
        .await?
        .remove(0);
    // create a notification
    let req = NotificationRequest::new("Test warning message!", NotificationLevel::Warn);
    client
        .images
        .create_notification(&group, &image.name, &req, &NotificationParams::default())
        .await?;
    // get the image's notifications
    let notification = user_client
        .images
        .get_notifications(&group, &image.name)
        .await?
        .remove(0);
    // fail to delete an image notification as a regular user
    let resp = user_client
        .images
        .delete_notification(&group, &image.name, &notification.id)
        .await;
    fail!(resp, 401);
    // fail to delete an image notification for an image that doesn't exist
    let resp = client
        .images
        .delete_notification(&group, "no-exists", &notification.id)
        .await;
    fail!(resp, 404);
    // fail to delete an image notification that doesn't exist
    let resp = client
        .images
        .delete_notification(&group, &image.name, &Uuid::new_v4())
        .await;
    fail!(resp, 404);
    Ok(())
}
