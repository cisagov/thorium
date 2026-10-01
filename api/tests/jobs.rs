//! Tests the Jobs routes in Thorium

use chrono::prelude::*;
use thorium::models::{ImageScaler, JobResets, ReactionListParams, ReactionStatus, Resources};
use thorium::test_utilities::{self, generators};
use thorium::{Error, is};

/// unwraps the status counts for a specific user and image
macro_rules! get_stats {
    ($stats:expr, $group:expr, $pipeline:expr, $stage:expr) => {
        if let Some(pipelines) = $stats.groups.get(&$group) {
            if let Some(stages) = pipelines.pipelines.get(&$pipeline) {
                if let Some(users) = stages.stages.get($stage) {
                    if let Some(statuses) = users.get(&"thorium".to_owned()) {
                        statuses
                    } else {
                        return Err(Error::new(format!(
                            "Failed == check because thorium not in {:#?}",
                            users
                        )));
                    }
                } else {
                    return Err(Error::new(format!(
                        "Failed == check because {} not in {:#?}",
                        $stage, stages
                    )));
                }
            } else {
                return Err(Error::new(format!(
                    "Failed == check because {} not in {:#?}",
                    $pipeline, pipelines.pipelines
                )));
            }
        } else {
            return Err(Error::new(format!(
                "Failed == check because {} not in {:#?}",
                $group, $stats.groups
            )));
        }
    };
}

#[tokio::test]
async fn proceed() -> Result<(), thorium::Error> {
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
    // get a timestamp before we create this reaction so we ensure its fits in the deadline stream
    let start = Utc::now();
    // make sure that we were able to create a reaction and our jobs
    let id = client.reactions.create(&req).await?;
    // make sure this external job is not in the deadline stream
    let end = start + chrono::Duration::hours(100000000);
    // attempt to claim a job for every stage of our reaction
    for stage in pipe.order.iter().flatten() {
        // register our test worker
        generators::worker(
            "cluster0", "node0", "proceed", &group, &pipe.name, stage, &client,
        )
        .await?;
        // make sure this stage updated the stage status counters correctly
        let stats = client.system.stats().await?;
        is!(get_stats!(stats, group, pipe_req.name, stage).created, 1);
        is!(get_stats!(stats, group, pipe_req.name, stage).running, 0);
        is!(get_stats!(stats, group, pipe_req.name, stage).completed, 0);
        is!(get_stats!(stats, group, pipe_req.name, stage).failed, 0);
        is!(get_stats!(stats, group, pipe_req.name, stage).sleeping, 0);
        is!(get_stats!(stats, group, pipe_req.name, stage).total, 1);
        // get our reactions data
        let react = client.reactions.get(&req.group, id.id).await?;
        // try to claim a job
        let job = client
            .jobs
            .claim(
                &req.group, &pipe.name, stage, "cluster0", "node0", "proceed", 1,
            )
            .await?;
        // check if any of our deadlines match our external job
        let deadlines = client
            .jobs
            .deadlines(ImageScaler::K8s, &start, &end, 10_000)
            .await?;
        let found = deadlines.iter().any(|item| item.job_id == job[0].id);
        is!(found, true);
        // make sure this job matches our reaction
        is!(&job, react);
        // build random stage logs
        let logs = generators::stage_logs();
        // make sure this stage updated the stage status counters correctly
        let stats = client.system.stats().await?;
        is!(get_stats!(stats, group, pipe_req.name, stage).created, 0);
        is!(get_stats!(stats, group, pipe_req.name, stage).running, 1);
        is!(get_stats!(stats, group, pipe_req.name, stage).completed, 0);
        is!(get_stats!(stats, group, pipe_req.name, stage).failed, 0);
        is!(get_stats!(stats, group, pipe_req.name, stage).sleeping, 0);
        is!(get_stats!(stats, group, pipe_req.name, stage).total, 1);
        // proceed with this stage
        client.jobs.proceed(&job[0], &logs, 12134).await?;
        // check our stage logs were correct
        client
            .reactions
            .logs(&group, &id.id, stage, &ReactionListParams::default())
            .await?;
        // make sure this stage updated the stage status counters correctly
        let stats = client.system.stats().await?;
        is!(get_stats!(stats, group, pipe_req.name, stage).created, 0);
        is!(get_stats!(stats, group, pipe_req.name, stage).running, 0);
        is!(get_stats!(stats, group, pipe_req.name, stage).completed, 1);
        is!(get_stats!(stats, group, pipe_req.name, stage).failed, 0);
        is!(get_stats!(stats, group, pipe_req.name, stage).sleeping, 0);
        is!(get_stats!(stats, group, pipe_req.name, stage).total, 1);
        // delete our worker
        generators::delete_worker("proceed", &client).await?;
    }
    Ok(())
}

#[tokio::test]
async fn error() -> Result<(), thorium::Error> {
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
    let req = generators::gen_reaction(&group, &pipe, None);
    // make sure that we were able to create a reaction and our jobs
    let id = client.reactions.create(&req).await?;
    // get our reactions data
    let react = client.reactions.get(&req.group, id.id).await?;
    // get the name of the first stage of this pipeline
    let stage = &pipe.order[0][0];
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // register our test worker
    generators::worker(
        "cluster0", "node0", "error", &group, &pipe.name, stage, &client,
    )
    .await?;
    // make sure this stage updated the stage status counters correctly
    let stats = client.system.stats().await?;
    is!(get_stats!(stats, group, pipe_req.name, stage).created, 1);
    is!(get_stats!(stats, group, pipe_req.name, stage).running, 0);
    is!(get_stats!(stats, group, pipe_req.name, stage).completed, 0);
    is!(get_stats!(stats, group, pipe_req.name, stage).failed, 0);
    is!(get_stats!(stats, group, pipe_req.name, stage).sleeping, 0);
    is!(get_stats!(stats, group, pipe_req.name, stage).total, 1);
    // try to claim a job for the first stage
    let job = client
        .jobs
        .claim(
            &req.group, &pipe.name, stage, "cluster0", "node0", "error", 1,
        )
        .await?;
    // make sure this stage updated the stage status counters correctly
    let stats = client.system.stats().await?;
    is!(get_stats!(stats, group, pipe_req.name, stage).created, 0);
    is!(get_stats!(stats, group, pipe_req.name, stage).running, 1);
    is!(get_stats!(stats, group, pipe_req.name, stage).completed, 0);
    is!(get_stats!(stats, group, pipe_req.name, stage).failed, 0);
    is!(get_stats!(stats, group, pipe_req.name, stage).sleeping, 0);
    is!(get_stats!(stats, group, pipe_req.name, stage).total, 1);
    // make sure this job matches our reaction
    is!(&job, react);
    // build random stage logs
    let logs = generators::stage_logs().code(137);
    // error out this stage
    client.jobs.error(&job[0].id, &logs).await?;
    // make sure this stage updated the stage status counters correctly
    let stats = client.system.stats().await?;
    is!(get_stats!(stats, group, pipe_req.name, stage).created, 0);
    is!(get_stats!(stats, group, pipe_req.name, stage).running, 0);
    is!(get_stats!(stats, group, pipe_req.name, stage).completed, 0);
    is!(get_stats!(stats, group, pipe_req.name, stage).failed, 1);
    is!(get_stats!(stats, group, pipe_req.name, stage).sleeping, 0);
    is!(get_stats!(stats, group, pipe_req.name, stage).total, 1);
    // delete our worker
    generators::delete_worker("error", &client).await?;
    Ok(())
}

#[tokio::test]
async fn empty() -> Result<(), thorium::Error> {
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
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // attempt to claim a job for every stage of our reaction
    for stage in pipe.order.iter().flatten() {
        // register our test worker
        generators::worker(
            "cluster0", "node0", "empty", &group, &pipe.name, stage, &client,
        )
        .await?;
        // try to claim a job
        let job = client
            .jobs
            .claim(&group, &pipe.name, stage, "cluster0", "node0", "empty", 1)
            .await?;
        // make sure this returned an empty list
        is!(job.is_empty(), true);
        // delete our worker
        generators::delete_worker("empty", &client).await?;
    }
    Ok(())
}

#[tokio::test]
async fn external_reset() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, true, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, None);
    // get a timestamp before we create this reaction so we ensure its fits in the deadline stream
    let start = Utc::now();
    // make sure this external job is not in the deadline stream
    let end = start + chrono::Duration::hours(100000000);
    // make sure that we were able to create a reaction and our jobs
    let id = client.reactions.create(&req).await?;
    // check if any of our deadlines match our external job
    let deadlines = client
        .jobs
        .deadlines(ImageScaler::K8s, &start, &end, 10_000)
        .await?;
    let found = deadlines.iter().any(|item| item.job_id == id.id);
    is!(found, false);
    // get our reactions data
    let react = client.reactions.get(&req.group, id.id).await?;
    // get our stage name
    let stage = &pipe.order[0][0];
    // register our test worker
    generators::worker_ext(
        "cluster0",
        "node0",
        "external_reset",
        &group,
        &pipe.name,
        stage,
        &client,
    )
    .await?;
    // try to claim a job for the first stage
    let job = client
        .jobs
        .claim(
            &req.group,
            &pipe.name,
            stage,
            "cluster0",
            "node0",
            "external_reset",
            1,
        )
        .await?;
    // make sure this job matches our reaction
    is!(&job, react);
    // build this list of jobs to reset
    let reset_req = JobResets::with_capacity(ImageScaler::K8s, "Test", 1).add(job[0].id);
    // reset this job
    client.jobs.bulk_reset(&reset_req).await?;
    // check if any of our deadlines match our external job
    let deadlines = client
        .jobs
        .deadlines(ImageScaler::K8s, &start, &end, 10_000)
        .await?;
    let found = deadlines.iter().any(|item| item.job_id == id.id);
    is!(found, false);
    // delete this reaction so we don't break any remaining tests
    client.reactions.delete(&req.group, &id.id).await?;
    // delete our worker
    generators::delete_worker_ext("external_reset", &client).await?;
    Ok(())
}

#[tokio::test]
async fn external_proceed() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::pipelines(&group, 1, true, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, None);
    // get a timestamp before we create this reaction so we ensure its fits in the deadline stream
    let start = Utc::now();
    // make sure that we were able to create a reaction and our jobs
    let id = client.reactions.create(&req).await?;
    // make sure this external job is not in the deadline stream
    let end = start + chrono::Duration::hours(100000000);
    let deadlines = client
        .jobs
        .deadlines(ImageScaler::K8s, &start, &end, 10_000)
        .await?;
    // check if any of our deadlines match our external job
    let found = deadlines.iter().any(|item| item.job_id == id.id);
    is!(found, false);
    // attempt to claim a job for every stage of our reaction
    for stage in pipe.order.iter().flatten() {
        // get our reactions data
        let react = client.reactions.get(&req.group, id.id).await?;
        // register our test worker
        generators::worker_ext(
            "cluster0",
            "node0",
            "external_proceed",
            &group,
            &pipe.name,
            stage,
            &client,
        )
        .await?;
        // try to claim a job
        let job = client
            .jobs
            .claim(
                &req.group,
                &pipe.name,
                stage,
                "cluster0",
                "node0",
                "external_proceed",
                1,
            )
            .await?;
        // make sure this job matches our reaction
        is!(&job, react);
        // build random stage logs
        let logs = generators::stage_logs();
        // proceed with this stage
        client.jobs.proceed(&job[0], &logs, 12134).await?;
        // delete our worker
        generators::delete_worker_ext("external_proceed", &client).await?;
    }
    Ok(())
}

#[tokio::test]
async fn checkpoint() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::gen_generator_pipe(&group, &client).await?;
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, None);
    // make sure that we were able to create a reaction and our jobs
    let id = client.reactions.create(&req).await?;
    // attempt to claim a job for every stage of our reaction
    for stage in pipe.order.iter().flatten().take(1) {
        for i in 0..3 {
            // get our reactions data
            let react = client.reactions.get(&req.group, id.id).await?;
            // register our test worker
            generators::worker(
                "cluster0",
                "node0",
                "checkpoint",
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
                    "checkpoint",
                    1,
                )
                .await?;
            // if this isn't the first loop then make sure our checkpoint was set
            if i != 0 {
                let check = vec![format!("checkpoint-{}", i - 1)];
                is!(jobs[0].args.kwargs.get("--checkpoint"), Some(&check));
            }
            // make sure this job matches our reaction
            is!(&jobs, react);
            // checkpoint this job
            let checkpoint = format!("checkpoint-{}", i);
            client.jobs.checkpoint(&jobs[0], checkpoint).await?;
            // if this isn't the last loop then sleep the generator
            if i < 2 {
                // checkpoint this job
                let checkpoint = format!("checkpoint-{}", i);
                // sleep this generator
                client.jobs.sleep(&jobs[0].id, checkpoint).await?;
            }
            // build random stage logs
            let logs = generators::stage_logs();
            // proceed with this stage
            client.jobs.proceed(&jobs[0], &logs, 12134).await?;
            // delete our worker
            generators::delete_worker_ext("checkpoint", &client).await?;
        }
    }
    Ok(())
}

#[tokio::test]
async fn sleep() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group to test reactions creation in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline
    let pipe_req = generators::gen_generator_pipe(&group, &client).await?;
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // Create a random reaction based on our pipeline request
    let req = generators::gen_reaction(&group, &pipe, None);
    // make sure that we were able to create a reaction and our jobs
    let id = client.reactions.create(&req).await?;
    // attempt to claim a job for every stage of our reaction
    for stage in pipe.order.iter().flatten().take(1) {
        for i in 0..3 {
            // get our reactions data
            let react = client.reactions.get(&req.group, id.id).await?;
            // register our test worker
            generators::worker(
                "cluster0", "node0", "sleep", &group, &pipe.name, stage, &client,
            )
            .await?;
            // try to claim a job
            let jobs = client
                .jobs
                .claim(
                    &req.group, &pipe.name, stage, "cluster0", "node0", "sleep", 1,
                )
                .await?;
            // if this isn't the first loop then make sure our checkpoint was set
            if i != 0 {
                let check = vec![format!("checkpoint-{}", i - 1)];
                is!(jobs[0].args.kwargs.get("--checkpoint"), Some(&check));
            }
            // make sure this job matches our reaction
            is!(&jobs, react);
            // if this isn't the last loop then sleep the generator
            if i < 2 {
                // checkpoint this job
                let checkpoint = format!("checkpoint-{}", i);
                // sleep this generator
                client.jobs.sleep(&jobs[0].id, checkpoint).await?;
            }
            // build random stage logs
            let logs = generators::stage_logs();
            // proceed with this stage
            client.jobs.proceed(&jobs[0], &logs, 12134).await?;
            // delete our worker
            generators::delete_worker("sleep", &client).await?;
        }
    }
    Ok(())
}

/// Duplicate concurrent proceed requests for one job only count that job once
///
/// `proceed` only executes the Redis pipeline that increments the reaction's
/// `current_stage_progress` if the job still has the status the route handler checked. Two
/// overlapping proceed requests for the SAME job (an agent retry racing the original) therefore
/// can't both count the job, so a stage of three jobs is not treated as complete after only two
/// of them finished.
///
/// The race is probabilistic, so this races many reactions and checks that none of them
/// completes while one of its three jobs is still running.
#[tokio::test]
#[ignore = "probabilistic race between concurrent duplicate proceed requests"]
async fn duplicate_concurrent_proceed_counts_job_once() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // create a group to test proceeding in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline with a single stage of three images
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // get the three images in our only stage
    let images = pipe.order[0].clone();
    // register a worker for each image in our stage
    for (i, image) in images.iter().enumerate() {
        generators::worker(
            "cluster0",
            "node0",
            &format!("c4_04_{i}"),
            &group,
            &pipe.name,
            image,
            &client,
        )
        .await?;
    }
    // race duplicate proceeds against many reactions to widen the race window
    for _ in 0..25 {
        // build a random reaction based on our pipeline request
        let req = generators::gen_reaction(&group, &pipe, None);
        // create the reaction and its three jobs
        let id = client.reactions.create(&req).await?;
        // claim the job for each image in our stage
        let mut jobs = Vec::with_capacity(images.len());
        for (i, image) in images.iter().enumerate() {
            let mut claimed = client
                .jobs
                .claim(
                    &group,
                    &pipe.name,
                    image,
                    "cluster0",
                    "node0",
                    &format!("c4_04_{i}"),
                    1,
                )
                .await?;
            jobs.push(claimed.remove(0));
        }
        // build random stage logs
        let logs = generators::stage_logs();
        // send two concurrent proceeds for the first job, at most one may succeed
        let _ = tokio::join!(
            client.jobs.proceed(&jobs[0], &logs, 1),
            client.jobs.proceed(&jobs[0], &logs, 1)
        );
        // complete the second job once
        client.jobs.proceed(&jobs[1], &logs, 1).await?;
        // the third job is still running so the reaction must not be completed
        let react = client.reactions.get(&group, id.id).await?;
        assert_ne!(
            react.status,
            ReactionStatus::Completed,
            "a duplicate proceed must not count a job twice and complete a stage early"
        );
        // complete the third job so this reaction finishes
        client.jobs.proceed(&jobs[2], &logs, 1).await?;
    }
    // delete our workers
    for i in 0..images.len() {
        generators::delete_worker(&format!("c4_04_{i}"), &client).await?;
    }
    Ok(())
}

/// An error and a proceed racing for the same job are not both applied
///
/// `error` (like `proceed`) only executes the Redis pipeline that marks the job Failed if the job
/// is still Running. An error request racing a proceed (or a duplicate error) for the same job
/// therefore can't both be applied, so a job is never both counted toward its stage and failed,
/// and its reaction is never failed by a job that also completed.
///
/// The race is probabilistic, so this races many reactions and checks that the error and the
/// proceed never both succeed.
#[tokio::test]
#[ignore = "probabilistic race between concurrent error and proceed"]
async fn concurrent_error_and_proceed_apply_once() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // register our test node
    generators::node("cluster0", "node0", Resources::default(), &client).await?;
    // create a group to test erroring in
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a random pipeline with a single stage of three images
    let pipe_req = generators::pipelines(&group, 1, false, &client)
        .await?
        .remove(0);
    // get the pipeline for this pipeline order
    let pipe = client.pipelines.get(&group, &pipe_req.name).await?;
    // get the first image in our only stage
    let image = pipe.order[0][0].clone();
    // register a worker for this image
    generators::worker(
        "cluster0",
        "node0",
        "adj_error",
        &group,
        &pipe.name,
        &image,
        &client,
    )
    .await?;
    // race an error against a proceed for many jobs to widen the race window
    for _ in 0..25 {
        // build a random reaction based on our pipeline request
        let req = generators::gen_reaction(&group, &pipe, None);
        // create the reaction and its jobs
        client.reactions.create(&req).await?;
        // claim the job for our image
        let mut claimed = client
            .jobs
            .claim(
                &group,
                &pipe.name,
                &image,
                "cluster0",
                "node0",
                "adj_error",
                1,
            )
            .await?;
        let job = claimed.remove(0);
        // build random stage logs
        let logs = generators::stage_logs();
        // send an error and a proceed for the same job at once, at most one may succeed
        let (errored, proceeded) = tokio::join!(
            client.jobs.error(&job.id, &logs),
            client.jobs.proceed(&job, &logs, 1)
        );
        assert!(
            !(errored.is_ok() && proceeded.is_ok()),
            "an error and a proceed for the same job must not both be applied"
        );
    }
    // delete our worker
    generators::delete_worker("adj_error", &client).await?;
    Ok(())
}
