use chrono::prelude::*;
use uuid::Uuid;

#[cfg(feature = "trace")]
use tracing::instrument;

use super::Error;
use crate::models::{
    Checkpoint, Deadline, GenericJob, HandleJobResponse, ImageScaler, JobResets, RunningJob,
    StageLogsAdd,
};
use crate::{send, send_build};

#[derive(Clone)]
pub struct Jobs {
    host: String,
    /// token to use for auth
    token: String,
    client: reqwest::Client,
}

impl Jobs {
    /// Creates a new jobs handler
    ///
    /// Instead of directly creating this handler you likely want to simply create a
    /// `thorium::Thorium` and use the handler within that instead.
    ///
    /// # Arguments
    ///
    /// * `host` - url/ip of the Thorium api
    /// * `token` - The token used for authentication
    /// * `client` - The reqwest client to use
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::client::Jobs;
    ///
    /// let client = reqwest::Client::new();
    /// let jobs = Jobs::new("http://127.0.0.1", "token", &client);
    /// ```
    #[must_use]
    pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
        // build basic route handler
        Jobs {
            host: host.to_owned(),
            token: token.to_owned(),
            client: client.clone(),
        }
    }
}

// only include blocking structs if the sync feature is enabled
cfg_if::cfg_if! {
    if #[cfg(feature = "sync")] {
        #[derive(Clone)]
        pub struct JobsBlocking {
            host: String,
            /// token to use for auth
            token: String,
            client: reqwest::Client,
        }

        impl JobsBlocking {
            /// creates a new blocking jobs handler
            ///
            /// Instead of directly creating this handler you likely want to simply create a
            /// `thorium::ThoriumBlocking` and use the handler within that instead.
            ///
            ///
            /// # Arguments
            ///
            /// * `host` - url/ip of the Thorium api
            /// * `token` - The token used for authentication
            /// * `client` - The reqwest client to use
            ///
            /// # Examples
            ///
            /// ```
            /// use thorium::client::JobsBlocking;
            ///
            /// let jobs = JobsBlocking::new("http://127.0.0.1", "token");
            /// ```
            pub fn new(host: &str, token: &str, client: &reqwest::Client) -> Self {
                // build basic route handler
                JobsBlocking {
                    host: host.to_owned(),
                    token: token.to_owned(),
                    client: client.clone(),
                }
            }
        }
    }
}

#[syncwrap::clone_impl]
impl Jobs {
    /// Claims [`GenericJob`]s from Thorium for a specific stage in a pipeline if any exist
    ///
    /// # Arguments
    ///
    /// * `group` - The group this pipeline is from
    /// * `pipeline` - The pipeline to claim jobs for
    /// * `stage` - The stage in the pipeline to claim a job for
    /// * `worker` - The name of the worker that is claiming jobs
    /// * `count` - The number of jobs to claim
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // claim a job from Thorium
    /// let jobs = thorium.jobs.claim("Corn", "Harvest", "CornHarvester", "prod", "node0", "esoteria", 1).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(name = "Thorium::Jobs::claim", skip(self), err(Debug))
    )]
    pub async fn claim(
        &self,
        group: &str,
        pipeline: &str,
        stage: &str,
        cluster: &str,
        node: &str,
        worker: &str,
        count: u64,
    ) -> Result<Vec<GenericJob>, Error> {
        // build url for claiming a job
        let url = format!(
            "{base}/api/jobs/claim/{group}/{pipeline}/{stage}/{cluster}/{node}/{worker}/{count}",
            base = &self.host,
            group = group,
            pipeline = pipeline,
            stage = stage,
            worker = worker,
            count = count
        );
        // build request
        let req = self.client.patch(&url).header("authorization", &self.token);
        // send this request and build a generic job from the response
        send_build!(self.client, req, Vec<GenericJob>)
    }

    /// Tell Thorium this job has succeeded and to proceed with it
    ///
    /// # Arguments
    ///
    /// * `job` - The job to proceed with
    /// * `logs` - The stdout/stderr logs to add for this stage
    /// * `runtime` - How long this job took in seconds
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{StageLogsAdd, JobHandleStatus};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // claim a job from Thorium
    /// let jobs = thorium.jobs.claim("Corn", "Harvest", "CornHarvester", "prod0", "node0", "esoteria", 1).await?;
    /// // execute all claimed jobs
    /// for job in jobs.iter() {
    ///     // do some work for this job and track execution time
    ///     // build the logs object we want to save
    ///     let logs = StageLogsAdd::default()
    ///         .logs(vec!("line1", "line2", "line3"));
    ///     // then proceed with it if it succeeds
    ///     thorium.jobs.proceed(job, &logs, 10).await?;
    /// }
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(
            name = "Thorium::Jobs::proceed",
            skip(self, job, logs),
            fields(job = job.id.to_string()),
            err(Debug)
        )
    )]
    pub async fn proceed(
        &self,
        job: &GenericJob,
        logs: &StageLogsAdd,
        runtime: u64,
    ) -> Result<HandleJobResponse, Error> {
        // build url for proceeding with a job
        let url = format!(
            "{base}/api/jobs/handle/{id}/proceed/{runtime}",
            base = &self.host,
            id = &job.id,
            runtime = runtime
        );
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(logs);
        // send this request and build a json value from the response
        send_build!(self.client, req, HandleJobResponse)
    }

    /// Tell Thorium this job has failed and to fail out the reaction
    ///
    /// # Arguments
    ///
    /// * `job` - The job to error out
    /// * `logs` - The stdout/stderr logs to add for this stage
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::StageLogsAdd};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // claim a job from Thorium
    /// let jobs = thorium.jobs.claim("Corn", "Harvest", "CornHarvester", "prod0", "node0", "esoteria", 1).await?;
    /// // execute all claimed jobs
    /// for job in jobs.iter() {
    ///     // do some work for this job
    ///     // build the logs object we want to save
    ///     let logs = StageLogsAdd::default()
    ///         .logs(vec!("line1", "line2", "error_line"));
    ///     // something went wrong error out
    ///     thorium.jobs.error(&job.id, &logs).await?;
    /// }
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(
            name = "Thorium::Jobs::error",
            skip_all,
            fields(job = id.to_string()),
            err(Debug)
        )
    )]
    pub async fn error(&self, id: &Uuid, logs: &StageLogsAdd) -> Result<HandleJobResponse, Error> {
        // build url for erroring out a job
        let url = format!(
            "{base}/api/jobs/handle/{id}/error",
            base = &self.host,
            id = &id
        );
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(logs);
        // send this request and build a json value from the response
        send_build!(self.client, req, HandleJobResponse)
    }

    /// Tell Thorium this generator should be slept instead of completed at the next complete
    ///
    /// # Arguments
    ///
    /// * `job` - The generator job to set as sleeping
    /// * `checkpoint` - An optional checkpoint to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::StageLogsAdd};
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // start with the generator's job id
    /// // (given by Thorium to the generator with the `--job` kwarg)
    /// let job_id = Uuid::new_v4();
    /// thorium.jobs.sleep(&job_id, "checkpoint-1").await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(
            name = "Thorium::Jobs::sleep",
            skip_all,
            fields(job = job_id.to_string()),
            err(Debug)
        )
    )]
    pub async fn sleep<T: Into<String>>(
        &self,
        job_id: &Uuid,
        checkpoint: T,
    ) -> Result<HandleJobResponse, Error> {
        // build url for sleeping a generator
        let url = format!(
            "{base}/api/jobs/handle/{job_id}/sleep",
            base = &self.host,
            job_id = job_id
        );
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(&Checkpoint {
                data: checkpoint.into(),
            });
        // send this request and build a json value from the response
        send_build!(self.client, req, HandleJobResponse)
    }

    /// Set a new checkpoint for this job
    ///
    /// This is mainly used by generators but it can be used by anything that can resume execution
    /// based on a string.
    ///
    /// # Arguments
    ///
    /// * `job` - The generator job to set as sleeping
    /// * `checkpoint` - The new checkpoint to set
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::{Thorium, models::StageLogsAdd};
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // claim a job from Thorium
    /// let jobs = thorium.jobs.claim("Corn", "Harvest", "CornHarvester", "prod0", "node0", "esoteria", 1).await?;
    /// // execute all claimed jobs
    /// for job in jobs.iter() {
    ///     // checkpoint this job
    ///     thorium.jobs.checkpoint(job, "checkpoint-1").await?;
    /// }
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(
            name = "Thorium::Jobs::checkpoint",
            skip_all,
            fields(job = job.id.to_string()),
            err(Debug)
        )
    )]
    pub async fn checkpoint<T: Into<String>>(
        &self,
        job: &GenericJob,
        checkpoint: T,
    ) -> Result<HandleJobResponse, Error> {
        // build url for exhausting a generator
        let url = format!(
            "{base}/api/jobs/handle/{id}/checkpoint",
            base = &self.host,
            id = &job.id,
        );
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(&Checkpoint {
                data: checkpoint.into(),
            });
        // send this request and build a json value from the response
        send_build!(self.client, req, HandleJobResponse)
    }

    /// List the deadlines between two timestamps up to a certain limit
    ///
    /// Due to how sorted sets work in redis if you have more deadlines then your limit it can
    /// be very difficult to crawl them. If your use case does require crawling them you should
    /// expect to miss deadlines when crawling as it is impossible to consistently page through
    /// them while new jobs are being added and claimed consistently.
    ///
    /// # Arguments
    ///
    /// * `start` - The timestamp to start listing deadlines at
    /// * `end` - The timestamp to stop listing deadlines at
    /// * `limit` - The max number of deadlines to retrieve
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::ImageScaler;
    /// use chrono::prelude::*;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // list the next 10k deadlines due in the next hour
    /// let start = Utc::now();
    /// let end = start + chrono::Duration::hours(1);
    /// let deadlines = thorium.jobs.deadlines(ImageScaler::K8s, &start, &end, 10_000).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(
            name = "Thorium::Jobs::deadlines",
            skip(self, start, end),
            fields(start = start.timestamp(), end = end.timestamp()),
            err(Debug)
        )
    )]
    pub async fn deadlines(
        &self,
        scaler: ImageScaler,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        limit: u64,
    ) -> Result<Vec<Deadline>, Error> {
        // build url for reading deadlines
        let url = format!(
            "{base}/api/jobs/deadlines/{scaler}/{start}/{end}",
            base = &self.host,
            scaler = scaler,
            start = start.timestamp(),
            end = end.timestamp()
        );
        // build request
        let req = self
            .client
            .get(&url)
            .header("authorization", &self.token)
            .query(&[("limit", limit)]);
        // send this request and build a vector of deadlines from the response
        send_build!(self.client, req, Vec<Deadline>)
    }

    /// List the running jobs between two timestamps up to a certain limit
    ///
    /// Due to how sorted sets work in redis if you have more running jobs then your limit it can
    /// be very difficult to crawl them. If your use case does require crawling them you should
    /// expect to miss running jobs when crawling as it is impossible to consistently page through
    /// them while new jobs are being added and removed.
    ///
    /// # Arguments
    ///
    /// * `start` - The timestamp to start listing deadlines at
    /// * `end` - The timestamp to stop listing deadlines at
    /// * `limit` - The max number of deadlines to retrieve
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::ImageScaler;
    /// use chrono::prelude::*;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // list the running jobs for the last hour to a minute after now
    /// // we do a minute after now so we don't miss any jobs that started running while this
    /// // request was being sent
    /// let start = Utc::now() - chrono::Duration::hours(1);
    /// let end = Utc::now() + chrono::Duration::minutes(1);
    /// let running = thorium.jobs.running(ImageScaler::K8s, &start, &end, 10_000).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(
            name = "Thorium::Jobs::running",
            skip(self, start, end),
            fields(
                start = start.timestamp(),
                end = end.timestamp()
            ),
            err(Debug)
        )
    )]
    pub async fn running(
        &self,
        scaler: ImageScaler,
        start: &DateTime<Utc>,
        end: &DateTime<Utc>,
        limit: u64,
    ) -> Result<Vec<RunningJob>, Error> {
        // build url for getting currently running jobs
        let url = format!(
            "{base}/api/jobs/bulk/running/{scaler}/{start}/{end}",
            base = &self.host,
            scaler = scaler,
            start = start.timestamp(),
            end = end.timestamp()
        );
        // build request
        let req = self
            .client
            .get(&url)
            .header("authorization", &self.token)
            .query(&[("limit", limit)]);
        // send this request and build a vector of running jobs from the response
        send_build!(self.client, req, Vec<RunningJob>)
    }

    /// Resets jobs in Thorium in bulk
    ///
    /// These jobs are normally reset because their worker was killed while executing this job.
    ///
    /// # Arguments
    ///
    /// * `IDs` - The job IDs to reset
    ///
    /// # Examples
    ///
    /// ```
    /// use thorium::Thorium;
    /// use thorium::models::{JobResets, ImageScaler};
    /// use chrono::prelude::*;
    /// use uuid::Uuid;
    /// # use thorium::Error;
    ///
    /// # async fn exec() -> Result<(), Error> {
    /// // create Thorium client
    /// let thorium = Thorium::build("http://127.0.0.1").token("<token>").build().await?;
    /// // build a job reset request
    /// let resets = JobResets::new(ImageScaler::K8s, "Servers were angry").add(Uuid::new_v4());
    /// thorium.jobs.bulk_reset(&resets).await?;
    /// # // allow test code to be compiled but don't unwrap as no API instance would be up
    /// # Ok(())
    /// # }
    /// # tokio_test::block_on(async {
    /// #    exec().await
    /// # });
    /// ```
    #[cfg_attr(
        feature = "trace",
        instrument(
            name = "Thorium::Jobs::bulk_reset",
            skip_all,
            fields(scaler = resets.scaler.as_str(), ids_len = resets.jobs.len()),
            err(Debug)
        )
    )]
    pub async fn bulk_reset(&self, resets: &JobResets) -> Result<reqwest::Response, Error> {
        // build url for getting currently running jobs
        let url = format!("{base}/api/jobs/bulk/reset", base = &self.host);
        // build request
        let req = self
            .client
            .post(&url)
            .header("authorization", &self.token)
            .json(&resets);
        // send this request
        send!(self.client, req)
    }
}
