//! Libvirt threadpool to run blocking virt calls in an async context

use std::num::NonZeroUsize;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use kanal::{AsyncSender, Receiver};
use serde_json::Value;
use tokio::sync::oneshot;
use tracing::{Level, event};
use virt::connect::Connect;
use virt::domain::Domain;

use crate::Error;

/// An async-friendly libvirt client with its own thread pool
/// that avoids blocking the async executor
#[derive(Clone)]
pub struct LibvirtClient {
    /// The thread pool dedicated to libvirt tasks
    pool: Arc<LibvirtPool>,
}

impl LibvirtClient {
    /// Create a new async-friendly libvirt client with a thread pool
    ///
    /// # Arguments
    ///
    /// * `uri` - The URI to connect to libvirt on
    /// * `pool_size` - The size of the OS thread pool, or the number of worker
    ///   threads to spawn
    pub fn new(uri: &str, pool_size: NonZeroUsize) -> Result<Self, Error> {
        Ok(Self {
            pool: Arc::new(LibvirtPool::new(uri, pool_size)?),
        })
    }

    /// Run a libvirt function using a connection in the thread pool
    ///
    /// # Arguments
    ///
    /// * `f` - The libvirt function to run in the thread pool
    pub async fn with_conn<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Connect) -> R + Send + 'static,
        R: Send + 'static,
    {
        self.pool.run(f).await
    }

    /// Run a libvirt function using a given reference to a domain
    ///
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain reference to use to connect
    pub async fn with_domain<F, R>(&self, domain: &Domain, f: F) -> R
    where
        F: FnOnce(&Domain) -> R + Send + 'static,
        R: Send + 'static,
    {
        let domain = domain.clone();
        self.pool.run(move |_conn| f(&domain)).await
    }

    /// Send a QEMU agent command to the given domain and deserialize
    /// its response to a `serde_json` [`Value`]
    ///
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain reference to use to connect
    /// * `cmd` - The cmd raw string to send to the QEMU agent
    /// * `timeout` - The command timeout
    /// * `flags` - Flags to send to the QEMU agent
    pub async fn agent_cmd<T>(
        &self,
        domain: &Domain,
        cmd: T,
        timeout: i32,
        flags: u32,
    ) -> Result<Value, Error>
    where
        T: Into<String>,
    {
        let domain = domain.clone();
        let cmd = cmd.into();
        // send the agent command
        let response_raw = self
            .pool
            .run(move |_conn| {
                let cmd = &cmd;
                domain.qemu_agent_command(cmd, timeout, flags)
            })
            .await
            .map_err(Error::from)?;
        // attempt to parse the JSON response
        serde_json::from_str(&response_raw).map_err(Error::from)
    }
}

/// A single job to run in the thread pool
type LibvirtJob = Box<dyn FnOnce(&Connect) + Send + 'static>;

/// A thread pool allocated strictly for libvirt tasks, allowing
/// the async executor to continue running while blocking libvirt
/// queries are running
///
/// Each thread in the pool owns a connection to libvirt
struct LibvirtPool {
    /// The channel to send jobs to the thread pool on
    jobs_tx: AsyncSender<LibvirtJob>,
    /// The worker threads
    workers: Vec<JoinHandle<()>>,
}

impl LibvirtPool {
    /// Create a new OS thread pool for libvirt-related tasks
    ///
    /// # Arguments
    ///
    /// * `uri` - The URI to connect to libvirt on
    /// * `pool_size` - The number of worker threads to spawn
    pub fn new(uri: &str, pool_size: NonZeroUsize) -> Result<Self, Error> {
        let pool_size: usize = pool_size.into();
        // create an async channel for sending jobs with queue size
        // limited to 10 times the number of threads
        let (jobx_tx, jobs_rx) = kanal::bounded::<LibvirtJob>(pool_size * 10);
        // create 'n' workers
        let mut workers = Vec::with_capacity(pool_size);
        for i in 0..pool_size {
            // each worker gets its own libvirt connection, jobs_rx, and OS thread
            let conn = Connect::open(Some(uri))?;
            let jobs_rx: Receiver<LibvirtJob> = jobs_rx.clone();
            let handle = thread::Builder::new()
                .name(format!("libvirt-worker-{i}"))
                .spawn(move || worker_loop(conn, jobs_rx))?;
            workers.push(handle);
        }
        Ok(Self {
            jobs_tx: jobx_tx.to_async(),
            workers,
        })
    }

    /// Request a libvirt function to be run on one of the threads in the
    /// in the thread pool
    ///
    /// # Arguments
    ///
    /// * `f` - The libvirt function to run in the thread pool
    pub async fn run<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Connect) -> R + Send + 'static,
        R: Send + 'static,
    {
        // use a simple tokio oneshot channel
        let (reply_tx, reply_rx) = oneshot::channel::<R>();
        let job: LibvirtJob = Box::new(move |conn| {
            let result = catch_unwind(AssertUnwindSafe(|| f(conn)));
            if let Ok(value) = result {
                // tokio oneshot send is sync and non-blocking; safe to
                // call from the worker thread with no runtime context.
                let _ = reply_tx.send(value);
            }
            // On panic: reply_tx drops, reply_rx.await returns Err below.
        });
        self.jobs_tx.send(job).await.expect(
            "Libvirt threadprool job channel was closed!! All worker threads may have exited",
        );
        reply_rx
            .await
            .expect("Libvirt threadpool worker panicked!!")
    }
}

impl Drop for LibvirtPool {
    fn drop(&mut self) {
        // close the channel to shutdown all workers
        self.jobs_tx
            .close()
            .expect("Fatal error closing the libvirt thread pool jobs channel!");
        // join our worker handles
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// The main worker loop: wait for jobs, do the job, and return the result
///
/// When the work channel is closed, close the libvirt connection
#[allow(clippy::needless_pass_by_value)]
fn worker_loop(conn: Connect, jobs_rx: Receiver<LibvirtJob>) {
    while let Ok(job) = jobs_rx.recv() {
        job(&conn);
    }
    let mut conn = conn;
    // let's explicitly close the connection to log any errors
    if let Err(e) = conn.close() {
        event!(
            Level::ERROR,
            "libvirt-worker: error closing connection: {e}"
        );
    }
}
