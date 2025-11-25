//! Tests the system routes in Thorium

use std::collections::HashSet;
use std::path::PathBuf;

use thorium::models::{
    HostPathWhitelistUpdate, ImageBanKind, PipelineBanKind, PipelineRequest, PipelineUpdate,
    SystemSettings, SystemSettingsResetParams, SystemSettingsUpdate, SystemSettingsUpdateParams,
    Volume, VolumeTypes,
};
use thorium::test_utilities::{self, generators};
use thorium::{contains, fail, is, is_not, unwrap_variant, vec_in_vec, Error};

#[serial_test::serial]
#[tokio::test]
async fn backup() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // reset settings to defaults
    client
        .system
        .reset_settings(&SystemSettingsResetParams::default().no_scan())
        .await?;
    // generate our groups/images/pipelines
    let (groups, images, pipes) = generators::gen_all(3, &client).await?;
    // get the backup
    let backup = client.system.backup().await?;
    // make sure the backup contains the default system settings that were set
    is!(&backup.settings, &SystemSettings::default());
    // make sure that our users list is not empty
    is!(backup.users.is_empty(), false);
    // make sure the backup contains all the groups/images/pipelines we created
    vec_in_vec!(&groups, &backup.groups);
    vec_in_vec!(&images, &backup.images);
    vec_in_vec!(&pipes, &backup.pipelines);
    Ok(())
}

#[serial_test::serial]
#[tokio::test]
async fn restore() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // generate our groups/images/pipelines
    let (groups, images, pipes) = generators::gen_all(3, &client).await?;
    // get our user
    let user = client.users.get("thorium").await?;
    // get the backup
    let backup = client.system.backup().await?;
    // restore our backup
    client.system.restore(&backup).await?;
    // get our user post restore
    let restored_user = client.users.get("thorium").await?;
    // make sure the user is correct
    is!(&user, &restored_user);
    // list the groups we just restored
    let resp = client.groups.list().page_size(500).details().exec().await?;
    // make sure all the group details we tried to restored are in our list
    vec_in_vec!(&groups, &resp.details);
    // crawl over the groups we restored and check to make sure all our images/pipelines were restored
    let mut restored_images = vec![];
    let mut restored_pipelines = vec![];
    for group in &groups {
        // list the images we just restored for this group
        let resp = client
            .images
            .list(&group.name)
            .page_size(500)
            .details()
            .exec()
            .await?;
        // add these images to our restored images list
        restored_images.extend(resp.details);
        // list the pipelines we just restored for this group
        let resp = client
            .pipelines
            .list(&group.name)
            .page_size(500)
            .details()
            .exec()
            .await?;
        // add these pipelines to our restored pipelines list
        restored_pipelines.extend(resp.details);
    }
    // make sure all the image/pipeline details we tried to restored are in our lists
    vec_in_vec!(&images, &restored_images);
    vec_in_vec!(&pipes, &restored_pipelines);
    Ok(())
}

#[serial_test::serial]
#[tokio::test]
async fn restore_create_reactions() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // generate our groups/images/pipelines
    let (groups, _, _) = generators::gen_all(3, &client).await?;
    // get the backup
    let backup = client.system.backup().await?;
    // restore our backup
    client.system.restore(&backup).await?;
    // try to create a reaction for each pipeline to make sure the restore worked
    for group in &groups {
        // list the pipelines we just restored for this group
        let resp = client
            .pipelines
            .list(&group.name)
            .page_size(500)
            .details()
            .exec()
            .await?;
        for pipe in &resp.details {
            // Create a random reaction based on our pipeline request
            let react_req = generators::gen_reaction(&group.name, pipe, None);
            client.reactions.create(&react_req).await?;
        }
    }
    Ok(())
}

#[serial_test::serial]
#[tokio::test]
async fn update_settings() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    let whitelist = ["/mnt/valid_path", "/mnt/valid_path2"];
    // create a new settings update
    let update = SystemSettingsUpdate::default()
        .reserved_cpu("10")
        .reserved_memory("1G")
        .reserved_storage("1G")
        .fairshare_cpu("5")
        .fairshare_memory("512M")
        .fairshare_storage("512M")
        .host_path_whitelist(HostPathWhitelistUpdate::default().add_paths(whitelist))
        .allow_unrestricted_host_paths(true);
    // build our params
    let default_params = SystemSettingsUpdateParams::default();
    // Update the settings
    client
        .system
        .update_settings(&update, &default_params)
        .await?;
    // Check that the settings were properly updated
    let settings = client.system.get_settings().await?;
    is!(&settings, &update);
    // update the settings again, removing one of the paths
    let update = SystemSettingsUpdate::default()
        .host_path_whitelist(HostPathWhitelistUpdate::default().remove_path(whitelist[0]));
    client
        .system
        .update_settings(&update, &default_params)
        .await?;
    // Check that the settings were properly updated
    let settings = client.system.get_settings().await?;
    is!(&settings, &update);
    // attempt to update settings as a regular user
    let user_client = generators::client(&client).await?;
    let resp = user_client
        .system
        .update_settings(&update, &default_params)
        .await;
    fail!(resp, 401);
    Ok(())
}

#[serial_test::serial]
#[tokio::test]
async fn reset_settings() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    let initially_valid_path = "/mnt/valid_path";
    // create a new settings update
    let update = SystemSettingsUpdate::default()
        .reserved_cpu("10")
        .reserved_memory("1G")
        .reserved_storage("1G")
        .fairshare_cpu("5")
        .fairshare_memory("512M")
        .fairshare_storage("512M")
        .host_path_whitelist(HostPathWhitelistUpdate::default().add_path(initially_valid_path))
        .allow_unrestricted_host_paths(true);
    // Update the settings
    client
        .system
        .update_settings(&update, &SystemSettingsUpdateParams::default().no_scan())
        .await?;
    // Check that the settings were properly updated
    let settings = client.system.get_settings().await?;
    is!(&settings, &update);
    // create user client
    let user_client = generators::client(&client).await?;
    // create group
    let group = generators::groups(1, &user_client).await?.remove(0).name;
    // create an image with a whitelisted host path
    let image_req = generators::gen_host_path(&group, initially_valid_path);
    user_client.images.create(&image_req).await?;
    // reset system settings, performing an automatic scan
    client
        .system
        .reset_settings(&SystemSettingsResetParams::default())
        .await?;
    // ensure settings have been reset
    let settings = client.system.get_settings().await?;
    is!(&settings, &SystemSettings::default());
    // make sure that image has a ban with that path now that it's no longer whitelisted
    let image = user_client.images.get(&group, &image_req.name).await?;
    is!(image.bans.len(), 1, "Ban added after settings reset");
    let ban = image.bans.values().next().unwrap();
    let ban = unwrap_variant!(&ban.ban_kind, ImageBanKind::InvalidHostPath);
    is!(ban.host_path, PathBuf::from(initially_valid_path));
    // attempt to reset settings as a regular user
    let user_client = generators::client(&client).await?;
    let resp = user_client
        .system
        .reset_settings(&SystemSettingsResetParams::default())
        .await;
    fail!(resp, 401);
    Ok(())
}

#[serial_test::serial]
#[tokio::test]
async fn update_host_path_whitelist() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // reset settings without scanning
    client
        .system
        .reset_settings(&SystemSettingsResetParams::default().no_scan())
        .await?;
    // get user client
    let user_client = generators::client(&client).await?;
    // create group
    let group = generators::groups(1, &user_client).await?.remove(0).name;
    let settings_update_default_params = SystemSettingsUpdateParams::default();
    // update with an invalid relative host path on the whitelist
    let update = SystemSettingsUpdate::default()
        .host_path_whitelist(HostPathWhitelistUpdate::default().add_path("invalid/relative"));
    let resp = client
        .system
        .update_settings(&update, &settings_update_default_params)
        .await;
    fail!(resp, 400, "relative traversal");
    // make sure the settings were not updated
    let settings = client.system.get_settings().await?;
    is_not!(&settings, &update);
    // update with a host path that is absolute but has relative traversal on the whitelist
    let update = SystemSettingsUpdate::default().host_path_whitelist(
        HostPathWhitelistUpdate::default().add_path("/absolute/../but/has/.."),
    );
    let resp = client
        .system
        .update_settings(&update, &settings_update_default_params)
        .await;
    fail!(resp, 400, "relative traversal");
    // add a path to the whitelist
    let whitelisted_path = "/allowed/path";
    let update = SystemSettingsUpdate::default()
        .host_path_whitelist(HostPathWhitelistUpdate::default().add_path(whitelisted_path));
    client
        .system
        .update_settings(&update, &settings_update_default_params)
        .await?;
    // create an image with that whitelisted host path
    let initially_valid_path = String::from(whitelisted_path) + "/here";
    let image_req = generators::gen_host_path(&group, &initially_valid_path);
    user_client.images.create(&image_req).await?;
    // change the settings to remove that host path from the whitelist
    client
        .system
        .update_settings(
            &SystemSettingsUpdate::default().host_path_whitelist(
                HostPathWhitelistUpdate::default().remove_path(whitelisted_path),
            ),
            &settings_update_default_params,
        )
        .await?;
    // make sure that image has a ban with that path now that it's no longer whitelisted
    let image = user_client.images.get(&group, &image_req.name).await?;
    is!(image.bans.len(), 1, "1 ban check after whitelist update");
    let ban = image.bans.values().next().unwrap();
    let ban = unwrap_variant!(&ban.ban_kind, ImageBanKind::InvalidHostPath);
    is!(ban.host_path, PathBuf::from(&initially_valid_path));
    Ok(())
}

#[serial_test::serial]
#[tokio::test]
async fn host_path_consistency_scan() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // reset settings to default
    client
        .system
        .reset_settings(&SystemSettingsResetParams::default().no_scan())
        .await?;
    // attempt to perform a successful scan
    client.system.consistency_scan().await?;
    // get user client
    let user_client = generators::client(&client).await?;
    // attempt to perform a scan as a regular user and expect a 401
    let resp = user_client.system.consistency_scan().await;
    fail!(resp, 401);
    // add a path to the whitelist
    let whitelisted_path = "/whitelisted/path";
    let update = SystemSettingsUpdate::default()
        .host_path_whitelist(HostPathWhitelistUpdate::default().add_path(whitelisted_path));
    client
        .system
        .update_settings(&update, &SystemSettingsUpdateParams::default().no_scan())
        .await?;
    // create group
    let group = generators::groups(1, &user_client).await?.remove(0).name;
    // create an image with paths on the whitelist
    let initially_valid_paths = [
        String::from(whitelisted_path) + "/here",
        String::from(whitelisted_path) + "/and/here",
    ];
    let mut image_req = generators::gen_host_path(&group, &initially_valid_paths[0]);
    image_req.volumes.push(Volume::new(
        "volume2",
        &initially_valid_paths[1],
        VolumeTypes::HostPath,
    ));
    user_client.images.create(&image_req).await?;
    // create a pipeline with that image
    let pipeline_name = "pipeline-ban-cs";
    let pipe_req = PipelineRequest::new(
        &group,
        pipeline_name,
        serde_json::json!(vec![vec![&image_req.name]]),
    );
    client.pipelines.create(&pipe_req).await?;
    // reset settings without scanning
    client
        .system
        .reset_settings(&SystemSettingsResetParams::default().no_scan())
        .await?;
    // run a scan
    client.system.consistency_scan().await?;
    // convert path Strings to PathBuf's
    let mut initially_valid_paths_pb: HashSet<PathBuf> =
        initially_valid_paths.iter().map(PathBuf::from).collect();
    // make sure that image has two bans with those paths now that they're no longer whitelisted
    let image = user_client.images.get(&group, &image_req.name).await?;
    is!(image.bans.len(), 2, "2 image bans after consistency scan");
    let ban = image.bans.values().next().unwrap();
    let ban = unwrap_variant!(&ban.ban_kind, ImageBanKind::InvalidHostPath);
    contains!(initially_valid_paths_pb, &ban.host_path);
    let ban = image.bans.values().next().unwrap();
    let ban_kind = unwrap_variant!(&ban.ban_kind, ImageBanKind::InvalidHostPath);
    contains!(initially_valid_paths_pb, &ban_kind.host_path);
    // make sure the pipeline has a ban
    let pipeline = user_client.pipelines.get(&group, &pipe_req.name).await?;
    is!(
        pipeline.bans.len(),
        1,
        "1 pipeline ban after consistency scan"
    );
    let ban = pipeline.bans.values().next().unwrap();
    let ban_kind = unwrap_variant!(&ban.ban_kind, PipelineBanKind::BannedImage);
    is!(ban_kind.image, image.name, "Pipeline ban is from image");
    // whitelist the second path but don't run an automatic scan
    let update = SystemSettingsUpdate::default().host_path_whitelist(
        HostPathWhitelistUpdate::default().add_path(&initially_valid_paths[1]),
    );
    client
        .system
        .update_settings(&update, &SystemSettingsUpdateParams::default().no_scan())
        .await?;
    // now scan, removing the ban with the whitelisted path
    client.system.consistency_scan().await?;
    // make sure that image now has only one ban
    initially_valid_paths_pb.remove(&PathBuf::from(&initially_valid_paths[1]));
    let image = user_client.images.get(&group, &image_req.name).await?;
    is!(image.bans.len(), 1, "1 ban after whitelist");
    let ban = image.bans.values().next().unwrap();
    let ban = unwrap_variant!(&ban.ban_kind, ImageBanKind::InvalidHostPath);
    contains!(initially_valid_paths_pb, &ban.host_path);
    // allow unrestricted host paths
    let update = SystemSettingsUpdate::default().allow_unrestricted_host_paths(true);
    client
        .system
        .update_settings(&update, &SystemSettingsUpdateParams::default().no_scan())
        .await?;
    // scan, removing the final ban
    client.system.consistency_scan().await?;
    // make sure the ban is gone
    let image = user_client.images.get(&group, &image_req.name).await?;
    is!(
        image.bans.len(),
        0,
        "Empty image bans after allow unrestricted"
    );
    // make sure the pipeline's ban is gone
    let pipeline = user_client.pipelines.get(&group, &pipe_req.name).await?;
    is!(
        pipeline.bans.len(),
        0,
        "Empty pipeline bans after allow unrestricted"
    );
    // create another image when all host paths are allowed
    let unrestricted_path = "/some/random/path";
    let image_req = generators::gen_host_path(&group, unrestricted_path);
    user_client.images.create(&image_req).await?;
    // add the image to the pipeline
    let mut order: Vec<Vec<String>> = pipeline.order.clone();
    order[0].push(image_req.name.clone());
    let pipe_update = PipelineUpdate::default().order(order);
    user_client
        .pipelines
        .update(&group, &pipeline.name, &pipe_update)
        .await?;
    // restrict host paths
    let update = SystemSettingsUpdate::default().allow_unrestricted_host_paths(false);
    client
        .system
        .update_settings(&update, &SystemSettingsUpdateParams::default().no_scan())
        .await?;
    // scan, adding a ban
    client.system.consistency_scan().await?;
    // make sure both images have bans again
    // image 1
    let image = user_client.images.get(&group, &image.name).await?;
    is!(image.bans.len(), 1, "1 ban after restricted 1");
    let ban = image.bans.values().next().unwrap();
    let ban_kind = unwrap_variant!(&ban.ban_kind, ImageBanKind::InvalidHostPath);
    contains!(initially_valid_paths_pb, &ban_kind.host_path);
    // image 2
    let new_image = user_client.images.get(&group, &image_req.name).await?;
    is!(new_image.bans.len(), 1, "1 ban after restricted 2");
    let ban = new_image.bans.values().next().unwrap();
    let ban_kind = unwrap_variant!(&ban.ban_kind, ImageBanKind::InvalidHostPath);
    is!(&PathBuf::from(unrestricted_path), &ban_kind.host_path);
    // make sure the pipeline has bans from both images
    let pipeline = user_client.pipelines.get(&group, &pipe_req.name).await?;
    is!(pipeline.bans.len(), 2, "2 pipeline ban after restricted");
    let mut pipeline_banned_images = Vec::with_capacity(2);
    for ban in pipeline.bans.values() {
        pipeline_banned_images
            .push(&unwrap_variant!(&ban.ban_kind, PipelineBanKind::BannedImage).image);
    }
    contains!(pipeline_banned_images, &&new_image.name);
    contains!(pipeline_banned_images, &&image.name);
    Ok(())
}
