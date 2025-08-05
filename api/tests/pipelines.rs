//! Tests the Pipelines routes in Thorium

use rand::{rng, seq::SliceRandom};
use thorium::models::{
    ImageBan, ImageBanKind, ImageBanUpdate, ImageUpdate, NotificationLevel, NotificationParams,
    NotificationRequest, PipelineBan, PipelineBanKind, PipelineBanUpdate, PipelineRequest,
    PipelineUpdate,
};
use thorium::test_utilities::{self, generators};
use thorium::{contains, fail, is, is_in, unwrap_variant, vec_in_vec, Error};
use uuid::Uuid;

#[tokio::test]
async fn create() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create the pipeline tests groups
    let group = generators::groups(1, &client).await?.remove(0).name;
    // generate a random pipeline request
    let pipe_req = generators::gen_pipe(&group, 20, false, &client).await?;
    // Create a test pipeline
    let resp = client.pipelines.create(&pipe_req).await?;
    is!(resp.status().as_u16(), 204);
    Ok(())
}

#[tokio::test]
async fn create_image_no_exist() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create the pipeline tests groups
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a pipeline with an image that doesn't exist
    let pipeline_name = "pipeline-no-exist-image";
    let pipe_req = PipelineRequest::new(
        &group,
        pipeline_name,
        serde_json::json!(vec![vec!["image-no-exist"]]),
    );
    // create it successfully
    let resp = client.pipelines.create(&pipe_req).await;
    fail!(resp, 404);
    Ok(())
}

#[tokio::test]
async fn create_conflict() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create the pipeline tests groups
    let group = generators::groups(1, &client).await?.remove(0).name;
    // generate a random pipeline request with no description
    let mut pipe_req = generators::gen_pipe(&group, 20, false, &client).await?;
    pipe_req.description = None;
    // Create a test pipeline
    let resp = client.pipelines.create(&pipe_req).await?;
    is!(resp.status().as_u16(), 204);
    pipe_req = pipe_req.description("This description should not be set on the existing pipeline");
    let resp = client.pipelines.create(&pipe_req).await;
    // check for a 409 conflict
    fail!(resp, 409);
    // check that the description for the existing pipeline was not set
    let pipeline = client
        .pipelines
        .get(&pipe_req.group, &pipe_req.name)
        .await?;
    is!(pipeline.description, None::<String>);
    Ok(())
}

#[tokio::test]
async fn create_banned_image() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create the pipeline tests groups
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create an image
    let image_req = generators::gen_image(&group);
    client.images.create(&image_req).await?;
    // add bans to the underlying image
    let image_bans = vec![
        ImageBan::new(ImageBanKind::generic("Test ban 1!")),
        ImageBan::new(ImageBanKind::generic("Test ban 2!")),
    ];
    let update =
        ImageUpdate::default().bans(ImageBanUpdate::default().add_bans(image_bans.clone()));
    client
        .images
        .update(&group, &image_req.name, &update)
        .await?;
    // create a pipeline with an image that is banned
    let pipeline_name = "pipeline-ban-new";
    let pipe_req = PipelineRequest::new(
        &group,
        pipeline_name,
        serde_json::json!(vec![vec![image_req.name]]),
    );
    // expect a 400
    let resp = client.pipelines.create(&pipe_req).await;
    fail!(resp, 400);
    Ok(())
}

#[tokio::test]
async fn get() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a pipeline and then get it
    let pipeline = generators::gen_pipe(&group, 20, false, &client).await?;
    let resp = client.pipelines.create(&pipeline).await?;
    is!(resp.status().as_u16(), 204);
    // get the pipeline and compare it
    let retrieved = client.pipelines.get(&group, &pipeline.name).await?;
    is!(retrieved, pipeline);
    Ok(())
}

#[tokio::test]
async fn update() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create the pipeline tests groups
    let group = generators::groups(1, &client).await?.remove(0).name;
    // generate a random pipeline request
    let pipe_req = generators::gen_pipe(&group, 20, false, &client).await?;
    // Create a test pipeline
    client.pipelines.create(&pipe_req).await?;
    // create an update request with a shuffled order
    let mut order: Vec<Vec<String>> = serde_json::from_value(pipe_req.order)?;
    order[0].shuffle(&mut rng());
    let pipe_update = PipelineUpdate::default()
        .order(order)
        .sla(86401)
        .description("Updated description")
        .bans(
            PipelineBanUpdate::default()
                .add_ban(PipelineBan::new(PipelineBanKind::generic("Test ban 1!")))
                .add_ban(PipelineBan::new(PipelineBanKind::generic("Test ban 2!"))),
        );
    // update the pipeline
    client
        .pipelines
        .update(&group, &pipe_req.name, &pipe_update)
        .await?;
    // check if the pipeline matches the update
    let retrieved = client.pipelines.get(&group, &pipe_req.name).await?;
    is!(retrieved, pipe_update);
    // update with clear variables
    let pipe_update = PipelineUpdate::default().clear_description();
    client
        .pipelines
        .update(&group, &pipe_req.name, &pipe_update)
        .await?;
    // check if the pipeline matches the update
    let retrieved = client.pipelines.get(&group, &pipe_req.name).await?;
    is!(retrieved, pipe_update);
    Ok(())
}

#[tokio::test]
async fn delete() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create the pipeline tests groups
    let group = generators::groups(1, &client).await?.remove(0).name;
    // generate a random pipeline request
    let pipe_req = generators::gen_pipe(&group, 20, false, &client).await?;
    // Create a test pipeline
    client.pipelines.create(&pipe_req).await?;
    // delete the pipeline
    client.pipelines.delete(&group, &pipe_req.name).await?;
    // make sure the pipeline was deleted
    let resp = client.pipelines.get(&group, &pipe_req.name).await;
    fail!(resp, 404);
    // TODO: test that all reactions were deleted?
    // TODO: test that all notifications were deleted; can't do that without specific
    // route to get notifications because we get a 404 when we try to get notifications
    // after deletion
    Ok(())
}

#[tokio::test]
async fn add_banned_image() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create the pipeline tests groups
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create an image
    let image_req = generators::gen_image(&group);
    client.images.create(&image_req).await?;
    // add bans to the underlying image
    let image_bans = vec![
        ImageBan::new(ImageBanKind::generic("Test ban 1!")),
        ImageBan::new(ImageBanKind::generic("Test ban 2!")),
    ];
    let update =
        ImageUpdate::default().bans(ImageBanUpdate::default().add_bans(image_bans.clone()));
    client
        .images
        .update(&group, &image_req.name, &update)
        .await?;
    // generate a random pipeline request
    let pipe_req = generators::gen_pipe(&group, 20, false, &client).await?;
    // Create a test pipeline
    client.pipelines.create(&pipe_req).await?;
    // add the banned image to the pipeline's order
    let order: Vec<Vec<String>> = serde_json::from_value(pipe_req.order.clone())?;
    let mut new_order = order.clone();
    new_order[0].push(image_req.name.clone());
    // create an update request with the banned image added
    let pipe_update = PipelineUpdate::default().order(new_order);
    // attempt to update the pipeline
    let resp = client
        .pipelines
        .update(&group, &pipe_req.name, &pipe_update)
        .await;
    // check that we got a 400 error
    fail!(resp, 400);
    // try again but as a sub stage
    let mut new_order = order.clone();
    new_order.push(vec![image_req.name.clone()]);
    // create an update request with the banned image added as a sub stage
    let pipe_update = PipelineUpdate::default().order(new_order);
    // attempt to update the pipeline
    let resp = client
        .pipelines
        .update(&group, &pipe_req.name, &pipe_update)
        .await;
    // check that we got a 400 error
    fail!(resp, 400);
    Ok(())
}

#[tokio::test]
async fn list() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 20 random pipelines
    let pipelines = generators::pipelines(&group, 20, false, &client).await?;
    // get the names of all the pipelines we have created
    let names: Vec<String> = pipelines.iter().map(|pipe| pipe.name.clone()).collect();
    // list the reactions we just created
    let mut cursor = client.pipelines.list(&group);
    cursor.next().await?;
    // make sure all the images we tried to create are in our list
    for pipe in names {
        is_in!(cursor.names, pipe);
    }
    Ok(())
}

#[tokio::test]
async fn list_details() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 20 random pipelines
    let pipelines = generators::pipelines(&group, 20, false, &client).await?;
    // list the pipelines we just created
    let mut cursor = client.pipelines.list(&group).details();
    cursor.next().await?;
    // make sure all the images we tried to create are in our list
    vec_in_vec!(&cursor.details, &pipelines);
    Ok(())
}

#[tokio::test]
async fn update_bans() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // generate a random pipeline request
    let pipe_req = generators::gen_pipe(&group, 20, false, &client).await?;
    // Create a test pipeline
    client.pipelines.create(&pipe_req).await?;
    // get the pipeline
    let pipeline = client.pipelines.get(&group, &pipe_req.name).await?;
    // update the pipeline with bans
    let mut bans = vec![
        PipelineBan::new(PipelineBanKind::generic("Test ban 1!")),
        PipelineBan::new(PipelineBanKind::image_ban(&pipeline.order[0][0])),
    ];
    let update =
        PipelineUpdate::default().bans(PipelineBanUpdate::default().add_bans(bans.clone()));
    client
        .pipelines
        .update(&group, &pipeline.name, &update)
        .await?;
    // successfully remove a ban from the image
    let update = PipelineUpdate::default().bans(
        PipelineBanUpdate::default().remove_ban(
            bans.pop()
                .ok_or_else(|| Error::new("Popped empty bans vec"))?
                .id,
        ),
    );
    client
        .pipelines
        .update(&group, &pipeline.name, &update)
        .await?;
    // get the pipeline and make sure the ban was removed
    let updated = client.pipelines.get(&group, &pipe_req.name).await?;
    is!(updated, update);
    // remove the second ban
    let update = PipelineUpdate::default().bans(
        PipelineBanUpdate::default().remove_ban(
            bans.pop()
                .ok_or_else(|| Error::new("Popped empty bans vec"))?
                .id,
        ),
    );
    client
        .pipelines
        .update(&group, &pipeline.name, &update)
        .await?;
    // get the pipeline and make sure the ban was removed
    let updated = client.pipelines.get(&group, &pipeline.name).await?;
    is!(updated, update);
    Ok(())
}

#[tokio::test]
async fn update_bans_bad() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // get user client
    let user_client = generators::client(&client).await?;
    // Create a group
    let group = generators::groups(1, &user_client).await?.remove(0).name;
    // generate a random pipeline request
    let pipe_req = generators::gen_pipe(&group, 20, false, &user_client).await?;
    // Create a test pipeline
    user_client.pipelines.create(&pipe_req).await?;
    // get the pipeline
    let pipeline = user_client.pipelines.get(&group, &pipe_req.name).await?;
    // update the pipeline with bans
    let mut bans = vec![
        PipelineBan::new(PipelineBanKind::generic("Test ban 1!")),
        PipelineBan::new(PipelineBanKind::image_ban(&pipeline.order[0][0])),
    ];
    let update =
        PipelineUpdate::default().bans(PipelineBanUpdate::default().add_bans(bans.clone()));
    client
        .pipelines
        .update(&group, &pipeline.name, &update)
        .await?;
    // attempt to add a ban as a non admin
    let resp = user_client
        .pipelines
        .update(&group, &pipeline.name, &update)
        .await;
    fail!(resp, 401);
    // attempt to remove a ban that doesn't exist
    let update =
        PipelineUpdate::default().bans(PipelineBanUpdate::default().remove_ban(Uuid::new_v4()));
    let resp = client
        .pipelines
        .update(&group, &pipeline.name, &update)
        .await;
    fail!(resp, 404);
    // attempt to add a ban that already exists
    let update =
        PipelineUpdate::default().bans(PipelineBanUpdate::default().add_ban(bans.pop().unwrap()));
    let resp = client
        .pipelines
        .update(&group, &pipeline.name, &update)
        .await;
    fail!(resp, 400);
    Ok(())
}

#[tokio::test]
async fn ban_from_image() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // get a user client
    let user_client = generators::client(&client).await?;
    // Create a group
    let group = generators::groups(1, &user_client).await?.remove(0).name;
    // create an image
    let image_req = generators::gen_image(&group);
    client.images.create(&image_req).await?;
    // create a pipeline with that image
    let pipeline_name = "pipeline-with-ban";
    let pipe_req = PipelineRequest::new(
        &group,
        pipeline_name,
        serde_json::json!(vec![vec![&image_req.name]]),
    );
    client.pipelines.create(&pipe_req).await?;
    // add bans to the underlying image
    let mut image_bans = vec![
        ImageBan::new(ImageBanKind::generic("Test ban 1!")),
        ImageBan::new(ImageBanKind::generic("Test ban 2!")),
    ];
    let update =
        ImageUpdate::default().bans(ImageBanUpdate::default().add_bans(image_bans.clone()));
    client
        .images
        .update(&group, &image_req.name, &update)
        .await?;
    // check that a ban was added to the pipeline
    let pipeline = client.pipelines.get(&group, pipeline_name).await?;
    is!(pipeline.bans.len(), 1);
    let ban = pipeline.bans.values().next().unwrap();
    let ban = unwrap_variant!(&ban.ban_kind, PipelineBanKind::BannedImage);
    is!(ban.image, image_req.name);
    // remove the first ban
    let update = ImageUpdate::default()
        .bans(ImageBanUpdate::default().remove_ban(image_bans.pop().unwrap().id));
    client
        .images
        .update(&group, &image_req.name, &update)
        .await?;
    // check that the ban is still there
    let pipeline = client.pipelines.get(&group, pipeline_name).await?;
    is!(pipeline.bans.len(), 1);
    let ban = pipeline.bans.values().next().unwrap();
    let ban = unwrap_variant!(&ban.ban_kind, PipelineBanKind::BannedImage);
    is!(ban.image, image_req.name);
    // remove the last ban
    let update = ImageUpdate::default()
        .bans(ImageBanUpdate::default().remove_ban(image_bans.pop().unwrap().id));
    client
        .images
        .update(&group, &image_req.name, &update)
        .await?;
    // check that the ban is removed
    let pipeline = client.pipelines.get(&group, pipeline_name).await?;
    is!(pipeline.bans.len(), 0);
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
    // create an image
    let image_req = generators::gen_image(&group);
    client.images.create(&image_req).await?;
    // create a pipeline with that image
    let pipeline_name = "pipeline-with-ban2";
    let pipe_req = PipelineRequest::new(
        &group,
        pipeline_name,
        serde_json::json!(vec![vec![&image_req.name]]),
    );
    client.pipelines.create(&pipe_req).await?;
    // update the pipeline with bans
    let generic_ban_msg = "Test ban 1!";
    let pipeline_generic_ban = PipelineBan::new(PipelineBanKind::generic(generic_ban_msg));
    let update = PipelineUpdate::default()
        .bans(PipelineBanUpdate::default().add_ban(pipeline_generic_ban.clone()));
    client
        .pipelines
        .update(&group, &pipe_req.name, &update)
        .await?;
    // add a ban to the pipeline by adding a ban to the image
    let image_ban = ImageBan::new(ImageBanKind::generic("Test ban 2!"));
    let update = ImageUpdate::default().bans(ImageBanUpdate::default().add_ban(image_ban.clone()));
    client
        .images
        .update(&group, &image_req.name, &update)
        .await?;
    // get the pipeline's bans
    let pipeline_bans = client.pipelines.get(&group, &pipe_req.name).await?.bans;
    // get the pipelines notifications and make sure that a notification was created for both bans
    let notifications_ban_ids = user_client
        .pipelines
        .get_notifications(&group, &pipe_req.name)
        .await?
        .into_iter()
        .map(|notification| notification.ban_id.unwrap())
        .collect::<Vec<Uuid>>();
    is!(notifications_ban_ids.len(), 2);
    let pipeline_image_ban = pipeline_bans
        .values()
        .find(|ban| matches!(&ban.ban_kind, PipelineBanKind::BannedImage(_)))
        .unwrap();
    contains!(notifications_ban_ids, &pipeline_image_ban.id);
    contains!(notifications_ban_ids, &pipeline_generic_ban.id);
    // remove the ban from the image
    let update = ImageUpdate::default().bans(ImageBanUpdate::default().remove_ban(image_ban.id));
    client
        .images
        .update(&group, &image_req.name, &update)
        .await?;
    // get the pipelines notifications and make sure that notification was automatically removed
    let notifications = user_client
        .pipelines
        .get_notifications(&group, &pipe_req.name)
        .await?;
    is!(notifications.len(), 1);
    is!(
        &pipeline_generic_ban.id,
        notifications[0].ban_id.as_ref().unwrap()
    );
    is!(notifications[0].msg, generic_ban_msg);
    Ok(())
}

#[tokio::test]
async fn create_notification() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // generate a random pipeline request
    let pipe_req = generators::gen_pipe(&group, 20, false, &client).await?;
    // Create a test pipeline
    client.pipelines.create(&pipe_req).await?;
    // create a pipeline notification
    let req = NotificationRequest::new("Test warning message!", NotificationLevel::Warn);
    client
        .pipelines
        .create_notification(&group, &pipe_req.name, &req, &NotificationParams::default())
        .await?;
    // make sure the pipeline notification is there
    let notifications = client
        .pipelines
        .get_notifications(&group, &pipe_req.name)
        .await?;
    is!(notifications.len(), 1);
    is!(notifications[0], req);
    is!(notifications[0].key.group, group);
    is!(notifications[0].key.pipeline, pipe_req.name);
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
    // generate a random pipeline request
    let pipe_req = generators::gen_pipe(&group, 20, false, &client).await?;
    // Create a test pipeline
    client.pipelines.create(&pipe_req).await?;
    // fail to create a pipeline notification as a regular user
    let req = NotificationRequest::new("Test warning message!", NotificationLevel::Warn);
    let resp = user_client
        .pipelines
        .create_notification(&group, &pipe_req.name, &req, &NotificationParams::default())
        .await;
    fail!(resp, 401);
    // fail to create a pipeline notification for a pipeline that doesn't exist
    let resp = client
        .pipelines
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
    // generate a random pipeline request
    let pipe_req = generators::gen_pipe(&group, 20, false, &client).await?;
    // Create a test pipeline
    client.pipelines.create(&pipe_req).await?;
    // create a notification
    let req = NotificationRequest::new("Test warning message!", NotificationLevel::Warn);
    client
        .pipelines
        .create_notification(&group, &pipe_req.name, &req, &NotificationParams::default())
        .await?;
    // get the pipeline notification
    let notification = client
        .pipelines
        .get_notifications(&group, &pipe_req.name)
        .await?
        .remove(0);
    // delete the notification
    client
        .pipelines
        .delete_notification(&group, &pipe_req.name, &notification.id)
        .await?;
    // check that the notification was deleted
    let notifications = client
        .pipelines
        .get_notifications(&group, &pipe_req.name)
        .await?;
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
    // set up a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &user_client)
        .await?
        .remove(0);
    // create a notification
    let req = NotificationRequest::new("Test warning message!", NotificationLevel::Warn);
    client
        .pipelines
        .create_notification(&group, &pipe_req.name, &req, &NotificationParams::default())
        .await?;
    // get the pipeline's notifications
    let notification = user_client
        .pipelines
        .get_notifications(&group, &pipe_req.name)
        .await?
        .remove(0);
    // fail to delete a pipeline notification as a regular user
    let resp = user_client
        .pipelines
        .delete_notification(&group, &pipe_req.name, &notification.id)
        .await;
    fail!(resp, 401);
    // fail to delete a pipeline notification for a pipeline that doesn't exist
    let resp = client
        .pipelines
        .delete_notification(&group, "no-exists", &notification.id)
        .await;
    fail!(resp, 404);
    // fail to delete a pipeline notification that doesn't exist
    let resp = client
        .pipelines
        .delete_notification(&group, &pipe_req.name, &Uuid::new_v4())
        .await;
    fail!(resp, 404);
    Ok(())
}
