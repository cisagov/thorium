//! Tests the Images routes in Thorium

use thorium::models::{
    GenericJobArgsUpdate, ImageBan, ImageBanKind, ImageBanUpdate, ImageUpdate, PipelineBan,
    PipelineBanKind, PipelineBanUpdate, PipelineRequest, PipelineUpdate, ReactionStatus,
    ReactionUpdate, Resources,
};
use thorium::test_utilities::{self, generators};
use thorium::{fail, is, is_empty, is_in, is_not_in, vec_in_vec, Error};
use uuid::Uuid;

#[tokio::test]
async fn create() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let react_req = generators::gen_reaction(&group, &pipe, None);
    let resp = client.reactions.create(&react_req).await?;
    // get the created reaction
    let created = client.reactions.get(&group, &resp.id).await?;
    // make sure our reaction request matches what was created
    is!(created, react_req);
    Ok(())
}

#[tokio::test]
async fn create_bulk() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // Create a random reactions in bulk based on our pipeline request
    let (_, resp) = generators::reactions(&group, 20, None, &client).await?;
    // make sure no errors were returned
    is_empty!(resp.errors);
    Ok(())
}

#[tokio::test]
async fn create_sub_reaction() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let react_req = generators::gen_reaction(&group, &pipe, None);
    let resp = client.reactions.create(&react_req).await?;
    // turn our old reaction request into a sub reaction and resubmit it
    let react_req = react_req.parent(resp.id);
    client.reactions.create(&react_req).await?;
    // make sure that our sub reaction counter incremented
    let reaction = client.reactions.get(&group, &resp.id).await?;
    is!(reaction.sub_reactions, 1);
    Ok(())
}

#[tokio::test]
async fn create_banned() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // add a ban to the pipeline
    let pipe_update = PipelineUpdate::default().bans(
        PipelineBanUpdate::default()
            .add_ban(PipelineBan::new(PipelineBanKind::generic("Generic ban!"))),
    );
    client
        .pipelines
        .update(&group, &pipe_req.name, &pipe_update)
        .await?;
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let react_req = generators::gen_reaction(&group, &pipe, None);
    let resp = client.reactions.create(&react_req).await;
    // make sure the reaction fails to spawn because of the ban
    fail!(resp, 400, "ban");
    // create a new pipeline whose image gets banned
    let image_req = generators::gen_image(&group);
    client.images.create(&image_req).await?;
    let pipeline_name = "pipeline-with-ban-react";
    let pipe_req = PipelineRequest::new(
        &group,
        pipeline_name,
        serde_json::json!(vec![vec![&image_req.name]]),
    );
    client.pipelines.create(&pipe_req).await?;
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
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let react_req = generators::gen_reaction(&group, &pipe, None);
    let resp = client.reactions.create(&react_req).await;
    // make sure the reaction fails to spawn because of the ban
    fail!(resp, 400, "ban");
    Ok(())
}

#[tokio::test]
async fn list() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 20 random reactions
    let (reactions, resp) = generators::reactions(&group, 20, None, &client).await?;
    // list the reactions we just created
    let mut cursor = client.reactions.list(&group, &reactions[0].pipeline);
    // get the next page of results
    cursor.next().await?;
    // make sure no errors were returned
    is_empty!(resp.errors);
    // make sure all the reactions we tried to create are in our list
    for id in &resp.created {
        is_in!(cursor.names, id.to_string());
    }
    Ok(())
}

#[tokio::test]
async fn list_details() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 20 random reactions
    let (reactions, _) = generators::reactions(&group, 20, None, &client).await?;
    // list the reactions we just created
    let mut cursor = client
        .reactions
        .list(&group, &reactions[0].pipeline)
        .details();
    cursor.next().await?;
    // make sure all the reactions we tried to create are in our list
    vec_in_vec!(&cursor.details, &reactions);
    Ok(())
}

#[tokio::test]
async fn list_status() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 20 random reactions
    let (reactions, resp) = generators::reactions(&group, 20, None, &client).await?;
    // list the reactions we just created
    let mut cursor =
        client
            .reactions
            .list_status(&group, &reactions[0].pipeline, &ReactionStatus::Created);
    cursor.next().await?;
    // make sure no errors were returned
    is_empty!(resp.errors);
    // make sure all the reactions we tried to create are in our list
    for id in &resp.created {
        is_in!(cursor.names, id.to_string());
    }
    Ok(())
}

#[tokio::test]
async fn list_status_details() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 20 random reactions
    let (reactions, _) = generators::reactions(&group, 20, None, &client).await?;
    // list the reactions we just created
    let mut cursor = client
        .reactions
        .list_status(&group, &reactions[0].pipeline, &ReactionStatus::Created)
        .details();
    cursor.next().await?;
    // make sure all the reactions we tried to create are in our list
    vec_in_vec!(&cursor.details, &reactions);
    Ok(())
}

#[tokio::test]
async fn list_tag() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 20 random reactions
    let (_, resp) = generators::reactions(&group, 20, Some("test"), &client).await?;
    // list the reactions we just created
    let mut cursor = client.reactions.list_tag(&group, "test");
    cursor.next().await?;
    // make sure no errors were returned
    is_empty!(resp.errors);
    // make sure all the reactions we tried to create are in our list
    for id in &resp.created {
        is_in!(cursor.names, id.to_string());
    }
    Ok(())
}

#[tokio::test]
async fn list_tag_details() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 20 random reactions
    let (reactions, _) = generators::reactions(&group, 20, Some("test_dets"), &client).await?;
    // list the reactions we just created
    let mut cursor = client.reactions.list_tag(&group, "test_dets").details();
    cursor.next().await?;
    // make sure all the reactions we tried to create are in our list
    vec_in_vec!(&cursor.details, &reactions);
    Ok(())
}

#[tokio::test]
async fn list_group_set() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 20 random reactions
    let (_, resp) = generators::reactions(&group, 20, Some("test"), &client).await?;
    // list the reactions we just created
    let mut cursor = client
        .reactions
        .list_group(&group, &ReactionStatus::Created);
    cursor.next().await?;
    // make sure no errors were returned
    is_empty!(resp.errors);
    // make sure all the reactions we tried to create are in our list
    for id in &resp.created {
        is_in!(cursor.names, id.to_string());
    }
    Ok(())
}

#[tokio::test]
async fn list_group_set_details() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 20 random reactions
    let (reactions, _) = generators::reactions(&group, 20, Some("test_dets"), &client).await?;
    // list the reactions we just created
    let mut cursor = client
        .reactions
        .list_group(&group, &ReactionStatus::Created)
        .details();
    cursor.next().await?;
    // make sure all the reactions we tried to create are in our list
    vec_in_vec!(&cursor.details, &reactions);
    Ok(())
}

#[tokio::test]
async fn list_sub() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 1 random parent reactions
    let (_, resp) = generators::reactions(&group, 1, None, &client).await?;
    // make sure no errors were returned
    is_empty!(resp.errors);
    // setup 20 random sub reactions
    let (_, sub_resp, _) = generators::sub_reactions(&group, 20, &resp.created[0], &client).await?;
    // list the reactions we just created
    let mut cursor = client.reactions.list_sub(&group, &resp.created[0]);
    cursor.next().await?;
    // make sure all the reactions we tried to create are in our list
    for created in &sub_resp {
        is_in!(cursor.names, created.id.to_string());
    }
    Ok(())
}

#[tokio::test]
async fn list_sub_details() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 1 random parent reactions
    let (_, resp) = generators::reactions(&group, 1, None, &client).await?;
    // make sure no errors were returned
    is_empty!(resp.errors);
    // setup 20 random sub reactions
    let (sub_reactions, _, _) =
        generators::sub_reactions(&group, 20, &resp.created[0], &client).await?;
    // list the reactions we just created
    let mut cursor = client
        .reactions
        .list_sub(&group, &resp.created[0])
        .details();
    cursor.next().await?;
    // make sure all the reactions we tried to create are in our list
    vec_in_vec!(&cursor.details, &sub_reactions);
    Ok(())
}

#[tokio::test]
async fn list_sub_status() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 1 random parent reactions
    let (_, resp) = generators::reactions(&group, 1, None, &client).await?;
    // make sure no errors were returned
    is_empty!(resp.errors);
    // setup 20 random sub reactions
    let (_, sub_resp, _) = generators::sub_reactions(&group, 20, &resp.created[0], &client).await?;
    // list the reactions we just created
    let mut cursor =
        client
            .reactions
            .list_sub_status(&group, &resp.created[0], &ReactionStatus::Created);
    cursor.next().await?;
    // make sure all the reactions we tried to create are in our list
    for created in &sub_resp {
        is_in!(cursor.names, created.id.to_string());
    }
    Ok(())
}

#[tokio::test]
async fn list_sub_status_details() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // setup 1 random parent reactions
    let (_, resp) = generators::reactions(&group, 1, None, &client).await?;
    // setup 20 random sub reactions
    let (sub_reactions, _, _) =
        generators::sub_reactions(&group, 20, &resp.created[0], &client).await?;
    // list the reactions we just created
    let mut cursor = client
        .reactions
        .list_sub_status(&group, &resp.created[0], &ReactionStatus::Created)
        .details();
    cursor.next().await?;
    // make sure all the reactions we tried to create are in our list
    vec_in_vec!(&cursor.details, &sub_reactions);
    Ok(())
}

#[tokio::test]
async fn claim() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, None);
    // make sure that we were able to create a reaction and our jobs
    let id = client.reactions.create(&req).await?;
    // attempt to claim a job for every stage of our reaction
    for stage in pipe.order.iter().flatten() {
        // register our test worker
        generators::worker(
            "cluster0", "node0", "claim", &group, &pipe.name, stage, &client,
        )
        .await?;
        // try to claim a job
        let jobs = client
            .jobs
            .claim(
                &req.group, &pipe.name, stage, "cluster0", "node0", "claim", 1,
            )
            .await?;
        // get our reactions data
        let react = client.reactions.get(&req.group, &id.id).await?;
        // make sure this job matches our reaction
        is!(&jobs, react);
        // build random stage logs
        let logs = generators::stage_logs();
        // complete this job
        client.jobs.proceed(&jobs[0], &logs, 2).await?;
        // delete our worker
        generators::delete_worker("claim", &client).await?;
    }
    // list the reactions for our pipeline that now has a status of completed
    let mut reactions =
        client
            .reactions
            .list_status(&group, &pipe.name, &ReactionStatus::Completed);
    reactions.next().await?;
    // make sure our reaction id is in the list
    is_in!(reactions.names, id.id.to_string());
    Ok(())
}

#[tokio::test]
async fn expires() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, None).tag("TestTag");
    // make sure that we were able to create a reaction and our jobs
    let id = client.reactions.create(&req).await?;
    // attempt to claim a job for every stage of our reaction
    for stage in pipe.order.iter().flatten() {
        // register our test worker
        generators::worker(
            "cluster0", "node0", "expires", &group, &pipe.name, stage, &client,
        )
        .await?;
        // try to claim a job
        let jobs = client
            .jobs
            .claim(
                &req.group, &pipe.name, stage, "cluster0", "node0", "expires", 1,
            )
            .await?;
        // get our reactions data
        let react = client.reactions.get(&req.group, &id.id).await?;
        // make sure this job matches our reaction
        is!(&jobs, react);
        // build random stage logs
        let logs = generators::stage_logs();
        // complete this job
        client.jobs.proceed(&jobs[0], &logs, 2).await?;
        // delete our worker
        generators::delete_worker("expires", &client).await?;
    }
    // list the reactions for our pipeline that now has a status of completed
    let mut reactions =
        client
            .reactions
            .list_status(&group, &pipe.name, &ReactionStatus::Completed);
    reactions.next().await?;
    // make sure our reaction id is in the list
    is_in!(reactions.names, id.id.to_string());
    // list the reactions we created our in our TestTag list
    let mut reactions = client.reactions.list_tag(&group, "TestTag");
    reactions.next().await?;
    // make sure our reaction id is in the list
    is_in!(reactions.names, id.id.to_string());
    // wait 10 seconds and then clean up old reactions
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    client.system.cleanup().await?;
    // list the reactions for our pipeline that now has a status of completed
    let mut reactions = client
        .reactions
        .list_status(&group, &pipe.name, &ReactionStatus::Completed)
        .exec()
        .await?;
    reactions.next().await?;
    // make sure our reaction list is empty
    is!(reactions.names.is_empty(), true);
    // list the reactions with our test tag
    let mut reactions = client.reactions.list_tag(&group, "TestTag");
    reactions.next().await?;
    // make sure our reaction list is empty
    is!(reactions.names.is_empty(), true);
    Ok(())
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn claim_generator() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::gen_generator_pipe(&group, &client).await?;
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, None);
    // make sure that we were able to create a reaction and our jobs
    let id = client.reactions.create(&req).await?;
    // get a flattened list of stages
    let stages: Vec<String> = pipe.order.clone().into_iter().flatten().collect();
    // attempt to claim and proceed with a generator 3 times
    for _ in 0..3 {
        // register our test worker
        generators::worker(
            "cluster0",
            "node0",
            "claim_generator",
            &group,
            &pipe.name,
            &stages[0],
            &client,
        )
        .await?;
        // try to claim a job
        let jobs = client
            .jobs
            .claim(
                &req.group,
                &pipe.name,
                &stages[0],
                "cluster0",
                "node0",
                "claim_generator",
                1,
            )
            .await?;
        // get our reactions data
        let react = client.reactions.get(&req.group, &id.id).await?;
        // make sure this job matches our reaction
        is!(&jobs, react);
        // build random stage logs
        let logs = generators::stage_logs();
        // sleep this generator
        client.jobs.sleep(&jobs[0].id, "checkpoint-1").await?;
        // complete this job
        client.jobs.proceed(&jobs[0], &logs, 2).await?;
        // try to claim the next stage of the reaction and make sure we don't get a job
        let next_stage = client
            .jobs
            .claim(
                &req.group,
                &pipe.name,
                &stages[1],
                "cluster0",
                "node0",
                "claim_generator",
                1,
            )
            .await?;
        // delete our worker
        generators::delete_worker("claim_generator", &client).await?;
        // make sure we failed to claim a job due to the previous stage failing
        is!(next_stage.len(), 0);
    }
    // register our test worker
    generators::worker(
        "cluster0",
        "node0",
        "claim_generator",
        &group,
        &pipe.name,
        &stages[0],
        &client,
    )
    .await?;
    // try to claim a generator job for the last time
    let jobs = client
        .jobs
        .claim(
            &req.group,
            &pipe.name,
            &stages[0],
            "cluster0",
            "node0",
            "claim_generator",
            1,
        )
        .await?;
    // get our reactions data
    let react = client.reactions.get(&req.group, &id.id).await?;
    // make sure this job matches our reaction
    is!(&jobs, react);
    // build random stage logs
    let logs = generators::stage_logs();
    // complete this job
    client.jobs.proceed(&jobs[0], &logs, 2).await?;
    // claim the final stage of this reaction and complete it
    let jobs = client
        .jobs
        .claim(
            &req.group,
            &pipe.name,
            &stages[1],
            "cluster0",
            "node0",
            "claim_generator",
            1,
        )
        .await?;
    // get our reactions data
    let react = client.reactions.get(&req.group, &id.id).await?;
    // make sure this job matches our reaction
    is!(&jobs, react);
    // build random stage logs
    let logs = generators::stage_logs();
    // complete this job
    client.jobs.proceed(&jobs[0], &logs, 2).await?;
    // delete our worker
    generators::delete_worker("claim_generator", &client).await?;
    // list the reactions for our pipeline that now has a status of completed
    let mut reactions =
        client
            .reactions
            .list_status(&group, &pipe.name, &ReactionStatus::Completed);
    reactions.next().await?;
    // make sure our reaction id is in the list
    is_in!(reactions.names, id.id.to_string());
    Ok(())
}

#[tokio::test]
async fn claim_generator_fail() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::gen_generator_pipe(&group, &client).await?;
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, None);
    // make sure that we were able to create a reaction and our jobs
    let id = client.reactions.create(&req).await?;

    // get a flattened list of stages
    let stages: Vec<String> = pipe.order.clone().into_iter().flatten().collect();
    // attempt to claim and proceed with a generator 3 times
    for _ in 0..3 {
        // register our test worker
        generators::worker(
            "cluster0",
            "node0",
            "claim_generator_fail",
            &group,
            &pipe.name,
            &stages[0],
            &client,
        )
        .await?;
        // try to claim a job
        let jobs = client
            .jobs
            .claim(
                &req.group,
                &pipe.name,
                &stages[0],
                "cluster0",
                "node0",
                "claim_generator_fail",
                1,
            )
            .await?;
        // get our reactions data
        let react = client.reactions.get(&req.group, &id.id).await?;
        // make sure this job matches our reaction
        is!(&jobs, react);
        // build random stage logs
        let logs = generators::stage_logs();
        // sleep this generator
        client.jobs.sleep(&jobs[0].id, "checkpoint-1").await?;
        // complete this job
        client.jobs.proceed(&jobs[0], &logs, 2).await?;
        // delete our worker
        generators::delete_worker("claim_generator_fail", &client).await?;
    }
    // register our test worker
    generators::worker(
        "cluster0",
        "node0",
        "claim_generator_fail",
        &group,
        &pipe.name,
        &stages[0],
        &client,
    )
    .await?;
    // try to claim a generator job for the last time
    let jobs = client
        .jobs
        .claim(
            &req.group,
            &pipe.name,
            &stages[0],
            "cluster0",
            "node0",
            "claim_generator_fail",
            1,
        )
        .await?;
    // get our reactions data
    let react = client.reactions.get(&req.group, &id.id).await?;
    // make sure this job matches our reaction
    is!(&jobs, react);
    // build random stage logs
    let logs = generators::stage_logs();
    // fail this job
    client.jobs.error(&jobs[0].id, &logs).await?;
    // try claim the final stage of this reaction and make sure it doesn't exist
    let jobs = client
        .jobs
        .claim(
            &req.group,
            &pipe.name,
            &stages[1],
            "cluster0",
            "node0",
            "claim_generator_fail",
            1,
        )
        .await?;
    // delete our worker
    generators::delete_worker("claim_generator_fail", &client).await?;
    // make sure we failed to claim a job due to the previous stage failing
    is!(jobs.len(), 0);
    Ok(())
}

#[tokio::test]
async fn claim_sub_reacts_status() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, None);
    // make sure that we were able to create a reaction and our jobs
    let resp = client.reactions.create(&req).await?;
    let (_, sub_resp, sub_pipe) = generators::sub_reactions(&group, 1, &resp.id, &client).await?;
    // get the id of the sub reaction we created
    let ids: Vec<Uuid> = sub_resp.into_iter().map(|created| created.id).collect();
    let id = ids[0].to_string();
    // list the reactions we just created
    let mut cursor = client
        .reactions
        .list_sub_status(&group, &resp.id, &ReactionStatus::Created);
    cursor.next().await?;
    // make sure our sub reaction is in the created status list
    is_in!(cursor.names, id);
    // complete all stages of this sub reaction
    for stage in sub_pipe.order.iter().flatten() {
        // register our test worker
        generators::worker(
            "cluster0",
            "node0",
            "claim_sub_reacts_status",
            &group,
            &pipe.name,
            stage,
            &client,
        )
        .await?;
        // try to claim a job for this sub reaction
        let jobs = client
            .jobs
            .claim(
                &req.group,
                &sub_pipe.name,
                stage,
                "cluster0",
                "node0",
                "claim_sub_reacts_status",
                1,
            )
            .await?;
        // build random stage logs
        let logs = generators::stage_logs();
        // complete this sub reactions job
        let status_resp = client.jobs.proceed(&jobs[0], &logs, 2).await?;
        // delete our worker
        generators::delete_worker("claim_sub_reacts_status", &client).await?;
        // make sure our sub reactions status is in the correct list
        let status = ReactionStatus::from(status_resp.status);
        // list the reactions we just created
        let mut cursor = client.reactions.list_sub_status(&group, &resp.id, &status);
        cursor.next().await?;
        // make sure our sub reaction is in the correct status list
        is_in!(cursor.names, id);
    }
    Ok(())
}

#[tokio::test]
async fn claim_sub_reacts() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::gen_generator_pipe(&group, &client).await?;
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, None);
    // make sure that we were able to create a reaction and our jobs
    let resp = client.reactions.create(&req).await?;
    // attempt to claim a job for every stage of our reaction
    for stage in pipe.order.iter().flatten() {
        // register our test worker
        generators::worker(
            "cluster0",
            "node0",
            "claim_sub_reacts",
            &group,
            &pipe.name,
            stage,
            &client,
        )
        .await?;
        // try to claim a job
        let jobs = client
            .jobs
            .claim(
                &req.group,
                &pipe.name,
                stage,
                "cluster0",
                "node0",
                "claim_sub_reacts",
                3,
            )
            .await?;
        // get our reactions data
        let react = client.reactions.get(&req.group, &resp.id).await?;
        // make sure this job matches our reaction
        is!(&jobs, react);
        // create 3 random sub reactions based on a random pipeline
        let (sub_reacts, _, sub_pipe) =
            generators::sub_reactions(&group, 3, &resp.id, &client).await?;
        // build random stage logs
        let logs = generators::stage_logs();
        // complete this job
        client.jobs.proceed(&jobs[0], &logs, 2).await?;
        // try to claim a job that shouldn't exist
        let jobs = client
            .jobs
            .claim(
                &req.group,
                &pipe.name,
                stage,
                "cluster0",
                "node0",
                "claim_sub_reacts",
                1,
            )
            .await?;
        // delete our worker
        generators::delete_worker("claim_sub_reacts", &client).await?;
        // make sure we failed to claim a job due to the sub reactions not being complete yet
        is!(jobs.len(), 0);
        // complete all sub reactions
        for _ in sub_reacts {
            // complete all stages of each sub reaction
            for stage in sub_pipe.order.iter().flatten() {
                // register our test worker
                generators::worker(
                    "cluster0",
                    "node0",
                    "claim_sub_reacts_sub",
                    &group,
                    &sub_pipe.name,
                    stage,
                    &client,
                )
                .await?;
                // try to claim a job for this sub reaction
                let jobs = client
                    .jobs
                    .claim(
                        &req.group,
                        &sub_pipe.name,
                        stage,
                        "cluster0",
                        "node0",
                        "claim_sub_reacts_sub",
                        1,
                    )
                    .await?;
                // build random stage logs
                let logs = generators::stage_logs();
                // complete this sub reactions job
                client.jobs.proceed(&jobs[0], &logs, 2).await?;
                // delete our worker
                generators::delete_worker("claim_sub_reacts_sub", &client).await?;
            }
        }
    }
    // list the reactions for our pipeline that now has a status of completed
    let mut reactions =
        client
            .reactions
            .list_status(&group, &pipe.name, &ReactionStatus::Completed);
    reactions.next().await?;
    // make sure our reaction id is in the list
    is_in!(reactions.names, resp.id.to_string());
    Ok(())
}

#[tokio::test]
async fn claim_sub_reacts_fail() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, None);
    // make sure that we were able to create a reaction and our jobs
    let resp = client.reactions.create(&req).await?;
    // attempt to claim a job for every stage of our reaction
    for stage in pipe.order.iter().flatten() {
        // register our test worker
        generators::worker(
            "cluster0",
            "node0",
            "claim_sub_reacts_fail",
            &group,
            &pipe.name,
            stage,
            &client,
        )
        .await?;
        // try to claim a job
        let jobs = client
            .jobs
            .claim(
                &req.group,
                &pipe.name,
                stage,
                "cluster0",
                "node0",
                "claim_sub_reacts_fail",
                1,
            )
            .await?;
        // get our reactions data
        let react = client.reactions.get(&req.group, &resp.id).await?;
        // make sure this job matches our reaction
        is!(&jobs, react);
        // create 3 random sub reactions based on a random pipeline
        let (sub_reacts, _, sub_pipe) =
            generators::sub_reactions(&group, 3, &resp.id, &client).await?;
        // build random stage logs
        let logs = generators::stage_logs();
        // complete this job
        client.jobs.proceed(&jobs[0], &logs, 2).await?;
        // try to claim a job that shouldn't exist yet
        let jobs = client
            .jobs
            .claim(
                &req.group,
                &pipe.name,
                stage,
                "cluster0",
                "node0",
                "claim_sub_reacts_fail",
                1,
            )
            .await?;
        // delete our worker
        generators::delete_worker("claim_sub_reacts_fail", &client).await?;
        // make sure we failed to claim a job due to the sub reactions not being failed yet
        is!(jobs.len(), 0);
        // fail all sub reactions
        for _ in sub_reacts {
            // fail the first stage of all sub reactions
            for stage in sub_pipe.order.iter().flatten().take(1) {
                // register our test worker
                generators::worker(
                    "cluster0",
                    "node0",
                    "claim_sub_reacts_fail_sub",
                    &group,
                    &sub_pipe.name,
                    stage,
                    &client,
                )
                .await?;
                // try to claim a job for this sub reaction
                let jobs = client
                    .jobs
                    .claim(
                        &req.group,
                        &sub_pipe.name,
                        stage,
                        "cluster0",
                        "node0",
                        "claim_sub_reacts_fail_sub",
                        1,
                    )
                    .await?;
                // build random stage logs
                let logs = generators::stage_logs();
                // complete this sub reactions job
                client.jobs.error(&jobs[0].id, &logs).await?;
                // delete our worker
                generators::delete_worker("claim_sub_reacts_fail_sub", &client).await?;
            }
        }
    }
    // list the reactions for our pipeline that now has a status of completed
    let mut reactions =
        client
            .reactions
            .list_status(&group, &pipe.name, &ReactionStatus::Completed);
    reactions.next().await?;
    // make sure our reaction id is in the list
    is_in!(reactions.names, resp.id.to_string());
    Ok(())
}

#[tokio::test]
async fn claim_sub_reacts_race() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, None);
    // make sure that we were able to create a reaction and our jobs
    let resp = client.reactions.create(&req).await?;
    // attempt to claim a job for every stage of our reaction
    for stage in pipe.order.iter().flatten() {
        // register our test worker
        generators::worker(
            "cluster0",
            "node0",
            "claim_sub_reacts_race",
            &group,
            &pipe.name,
            stage,
            &client,
        )
        .await?;
        // try to claim a job
        let jobs = client
            .jobs
            .claim(
                &req.group,
                &pipe.name,
                stage,
                "cluster0",
                "node0",
                "claim_sub_reacts_race",
                1,
            )
            .await?;
        // get our reactions data
        let react = client.reactions.get(&req.group, &resp.id).await?;
        // make sure this job matches our reaction
        is!(&jobs, react);
        // create 3 random sub reactions based on a random pipeline
        let (sub_reacts, _, sub_pipe) =
            generators::sub_reactions(&group, 3, &resp.id, &client).await?;
        // build random stage logs
        let logs = generators::stage_logs();
        // complete all sub reactions
        for _ in sub_reacts {
            // complete all stages of each sub reaction
            for stage in sub_pipe.order.iter().flatten() {
                // register our test worker
                generators::worker(
                    "cluster0",
                    "node0",
                    "claim_sub_reacts_race_sub",
                    &group,
                    &sub_pipe.name,
                    stage,
                    &client,
                )
                .await?;
                // try to claim a job for this sub reaction
                let jobs = client
                    .jobs
                    .claim(
                        &req.group,
                        &sub_pipe.name,
                        stage,
                        "cluster0",
                        "node0",
                        "claim_sub_reacts_race_sub",
                        1,
                    )
                    .await?;
                // build random stage logs
                let logs = generators::stage_logs();
                // complete this sub reactions job
                client.jobs.proceed(&jobs[0], &logs, 2).await?;
                // delete our worker
                generators::delete_worker("claim_sub_reacts_race_sub", &client).await?;
            }
        }
        // try to claim a job that shouldn't exist yet
        let check_jobs = client
            .jobs
            .claim(
                &req.group,
                &pipe.name,
                stage,
                "cluster0",
                "node0",
                "claim_sub_reacts_race",
                1,
            )
            .await?;
        // make sure we failed to claim a job due to the sub reactions not being complete yet
        is!(check_jobs.len(), 0);
        // complete this job
        client.jobs.proceed(&jobs[0], &logs, 2).await?;
        // delete our worker
        generators::delete_worker("claim_sub_reacts_race", &client).await?;
    }
    // list the reactions for our pipeline that now has a status of completed
    let mut reactions =
        client
            .reactions
            .list_status(&group, &pipe.name, &ReactionStatus::Completed);
    reactions.next().await?;
    // make sure our reaction id is in the list
    is_in!(reactions.names, resp.id.to_string());
    Ok(())
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn claim_sub_reacts_race_fail() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, None);
    // make sure that we were able to create a reaction and our jobs
    let resp = client.reactions.create(&req).await?;
    // attempt to claim a job for every stage of our reaction
    for stage in pipe.order.iter().flatten() {
        // register our test worker
        generators::worker(
            "cluster0",
            "node0",
            "claim_sub_reacts_race_fail",
            &group,
            &pipe.name,
            stage,
            &client,
        )
        .await?;
        // try to claim a job
        let jobs = client
            .jobs
            .claim(
                &req.group,
                &pipe.name,
                stage,
                "cluster0",
                "node0",
                "claim_sub_reacts_race_fail",
                1,
            )
            .await?;
        // get our reactions data
        let react = client.reactions.get(&req.group, &resp.id).await?;
        // make sure this job matches our reaction
        is!(&jobs, react);
        // create 3 random sub reactions based on a random pipeline
        let (sub_reacts, _, sub_pipe) =
            generators::sub_reactions(&group, 3, &resp.id, &client).await?;
        // build random stage logs
        let logs = generators::stage_logs();
        // complete all sub reactions
        for _ in sub_reacts {
            // complete all stages of each sub reaction
            for stage in sub_pipe.order.iter().flatten().take(1) {
                // register our test worker
                generators::worker(
                    "cluster0",
                    "node0",
                    "claim_sub_reacts_race_fail_sub",
                    &group,
                    &pipe.name,
                    stage,
                    &client,
                )
                .await?;
                // try to claim a job for this sub reaction
                let jobs = client
                    .jobs
                    .claim(
                        &req.group,
                        &sub_pipe.name,
                        stage,
                        "cluster0",
                        "node0",
                        "claim_sub_reacts_race_fail_sub",
                        1,
                    )
                    .await?;
                // build random stage logs
                let logs = generators::stage_logs();
                // complete this sub reactions job
                client.jobs.error(&jobs[0].id, &logs).await?;
                // delete our worker
                generators::delete_worker("claim_sub_reacts_race_fail_sub", &client).await?;
            }
        }
        // register our test worker
        generators::worker(
            "cluster0",
            "node0",
            "claim_sub_reacts_race_fail_empty",
            &group,
            &pipe.name,
            stage,
            &client,
        )
        .await?;
        // try to claim a job that shouldn't exist yet
        let check_jobs = client
            .jobs
            .claim(
                &req.group,
                &pipe.name,
                stage,
                "cluster0",
                "node0",
                "claim_sub_reacts_race_fail_empty",
                1,
            )
            .await?;
        // delete our worker
        generators::delete_worker("claim_sub_reacts_race_fail_empty", &client).await?;
        // make sure we failed to claim a job due to the sub reactions not being complete yet
        is!(check_jobs.len(), 0);
        // complete this job
        client.jobs.proceed(&jobs[0], &logs, 2).await?;
        // delete our worker
        generators::delete_worker("claim_sub_reacts_race_fail", &client).await?;
    }
    // list the reactions for our pipeline that now has a status of completed
    let mut reactions =
        client
            .reactions
            .list_status(&group, &pipe.name, &ReactionStatus::Completed);
    reactions.next().await?;
    // make sure our reaction id is in the list
    is_in!(reactions.names, resp.id.to_string());
    Ok(())
}

#[tokio::test]
async fn update() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let react_req = generators::gen_reaction(&group, &pipe, None);
    let create = client.reactions.create(&react_req).await?;
    // build an update for this reaction
    let update = ReactionUpdate::default().tag("wooooooot").arg(
        &pipe.order[0][0],
        GenericJobArgsUpdate::default()
            .positionals(vec!["1", "2", "3"])
            .kwarg("--new", vec!["ImANewArg"]),
    );
    // update this reaction
    let reaction = client.reactions.update(&group, &create.id, &update).await?;
    // make sure our reaction was applied
    is!(reaction, update);
    Ok(())
}

#[tokio::test]
async fn delete() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, Some("CornTag"));
    // make sure that we were able to create a reaction and our jobs
    let id = client.reactions.create(&req).await?;
    // list the reactions for our pipeline
    let mut reactions = client.reactions.list(&group, &pipe.name);
    reactions.next().await?;
    // make sure our reaction id is in this list
    is_in!(reactions.names, id.id.to_string());
    // list the reactions for this group
    let mut reactions = client
        .reactions
        .list_group(&group, &ReactionStatus::Created);
    reactions.next().await?;
    // make sure our reaction id is in this list
    is_in!(reactions.names, id.id.to_string());
    // list the reactions for our pipeline that now has a status of created
    let mut reactions = client
        .reactions
        .list_status(&group, &pipe.name, &ReactionStatus::Created);
    reactions.next().await?;
    // make sure our reaction id is in this list
    is_in!(reactions.names, id.id.to_string());
    // list the reactions with a specific tag
    let mut reactions = client.reactions.list_tag(&group, "CornTag");
    reactions.next().await?;
    // make sure our reaction id is in this list
    is_in!(reactions.names, id.id.to_string());
    // delete this reaction
    client.reactions.delete(&group, &id.id).await?;
    // list the reactions for our pipeline
    let mut reactions = client.reactions.list(&group, &pipe.name);
    reactions.next().await?;
    // make sure our reaction id is not in this list
    is_not_in!(reactions.names, id.id.to_string());
    // list the reactions for our pipeline
    let mut reactions = client
        .reactions
        .list_group(&group, &ReactionStatus::Created);
    reactions.next().await?;
    // make sure our reaction id is not in this list
    is_not_in!(reactions.names, id.id.to_string());
    // list the reactions for our pipeline that now has a status of created
    let mut reactions = client
        .reactions
        .list_status(&group, &pipe.name, &ReactionStatus::Created);
    reactions.next().await?;
    // make sure our reaction id is not in this list
    is_not_in!(reactions.names, id.id.to_string());
    // list the reactions with a specific tag
    let mut reactions = client.reactions.list_tag(&group, "CornTag");
    reactions.next().await?;
    // make sure our reaction id is not in this list
    is_not_in!(reactions.names, id.id.to_string());
    Ok(())
}

#[tokio::test]
async fn ephemeral() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let react_req =
        generators::gen_reaction(&group, &pipe, None).buffer("test-file", "I am a test file");
    let resp = client.reactions.create(&react_req).await?;
    // download the ephemeral file and make sure its correct
    let download = client
        .reactions
        .download_ephemeral(&group, &resp.id, "test-file")
        .await?;
    is!(download, "I am a test file");
    // get the created reaction
    let created = client.reactions.get(&group, &resp.id).await?;
    // make sure our reaction request matches what was created
    is!(created, react_req);
    // complete this reaction to make sure purging ephemeral files works
    for stage in pipe.order.iter().flatten() {
        // register our test worker
        generators::worker(
            "cluster0",
            "node0",
            "ephemeral",
            &group,
            &pipe.name,
            stage,
            &client,
        )
        .await?;
        // try to claim a job
        let jobs = client
            .jobs
            .claim(
                &group,
                &pipe.name,
                stage,
                "cluster0",
                "node0",
                "ephemeral",
                1,
            )
            .await?;
        // get our reactions data
        let react = client.reactions.get(&group, &created.id).await?;
        // make sure this job matches our reaction
        is!(&jobs, react);
        // build random stage logs
        let logs = generators::stage_logs();
        // complete this job
        client.jobs.proceed(&jobs[0], &logs, 2).await?;
        // delete our worker
        generators::delete_worker("ephemeral", &client).await?;
    }
    Ok(())
}

#[tokio::test]
async fn ephemeral_fail() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random reaction based on our pipeline request
    let react_req =
        generators::gen_reaction(&group, &pipe, None).buffer("test-file", "I am a test file");
    let resp = client.reactions.create(&react_req).await?;
    // get the created reaction
    let created = client.reactions.get(&group, &resp.id).await?;
    // make sure our reaction request matches what was created
    is!(created, react_req);
    // get the next stage so we can claim the job and then fail it
    let stage = pipe.order.iter().flatten().next().unwrap();
    // register our test worker
    generators::worker(
        "cluster0",
        "node0",
        "ephemeral_fail",
        &group,
        &pipe.name,
        stage,
        &client,
    )
    .await?;
    // try to claim a job for this sub reaction
    let jobs = client
        .jobs
        .claim(
            &group,
            &pipe.name,
            stage,
            "cluster0",
            "node0",
            "ephemeral_fail",
            1,
        )
        .await?;
    // build random stage logs
    let logs = generators::stage_logs();
    // complete this sub reactions job
    client.jobs.error(&jobs[0].id, &logs).await?;
    // delete our worker
    generators::delete_worker("ephemeral_fail", &client).await?;
    Ok(())
}

#[tokio::test]
async fn parent_ephemeral() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // Create a random parent reaction based on our pipeline request
    let react_req = generators::gen_reaction(&group, &pipe, None)
        .buffer("parent-file", "I am a parent test file");
    let resp = client.reactions.create(&react_req).await?;
    // Create a random child reaction based on our pipeline request
    let react_req = generators::gen_reaction(&group, &pipe, None)
        .parent(resp.id)
        .buffer("child-file", "I am a child test file");
    let resp = client.reactions.create(&react_req).await?;
    // download the child ephemeral file and make sure its correct
    let download = client
        .reactions
        .download_ephemeral(&group, &resp.id, "child-file")
        .await?;
    is!(download, "I am a child test file");
    // get the created reaction
    let created = client.reactions.get(&group, &resp.id).await?;
    // make sure our reaction request matches what was created
    is!(created, react_req);
    // make sure our parent ephemeral file was propagated
    is!(
        created.parent_ephemeral.get("parent-file"),
        created.parent.as_ref()
    );
    // download the child ephemeral file and make sure its correct
    let download = client
        .reactions
        .download_ephemeral(&group, &created.parent.unwrap(), "parent-file")
        .await?;
    is!(download, "I am a parent test file");
    Ok(())
}
