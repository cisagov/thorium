//! Shared objects and methods across all requests
use axum::extract::FromRef;
use bb8_redis::{bb8::Pool, RedisConnectionManager};
use elasticsearch::Elasticsearch;
use lettre::message::header::ContentType;
use lettre::message::{IntoBody, Mailbox};
use lettre::transport::smtp::authentication::Credentials;
use lettre::{AsyncSmtpTransport, AsyncTransport, Tokio1Executor};
use regex::RegexSet;
use std::sync::Arc;
use tokio::fs;

use super::s3::S3;
use crate::info;
use crate::models::backends::setup::{self, Scylla};
use crate::utils::ApiError;
use crate::{conf::Conf, error};

/// Tries to execute a future 10 times with a custom timeout
///
/// # Arguments
///
/// * `future` - The future to try to complete
/// * `timeout` - How long to wait for each attempt to complete
macro_rules! retry {
    ($future:expr, $timeout:expr, $name:expr, $config:expr) => {{
        // setup a counter variable at 0 to track how many attempts have been made
        let mut i = 0;
        // loop and try to complete this future
        loop {
            match tokio::time::timeout(std::time::Duration::from_secs($timeout), $future).await {
                //    // the future completed so return the result
                Ok(res) => break res,
                // the future failed so try again if we have failed less then 10 times or panic
                Err(err) => {
                    // log this error
                    error!(
                        $config.thorium.tracing.local.level,
                        format!(
                            "Future {} failed to complete in {} seconds. Restarting!",
                            $name, $timeout
                        )
                    );
                    if i == 9 {
                        // we failed 10 times so panic
                        panic!("{:#?}", err)
                    } else {
                        // increment i and try again
                        i += 1;
                        continue;
                    }
                }
            }
        }
    }};
}

/// A client for sending emails from Thorium
pub struct EmailClient {
    /// The address to send emails from
    from: Mailbox,
    /// The email client to use
    client: AsyncSmtpTransport<Tokio1Executor>,
    /// The approved emails regexes for users in Thorium
    pub approved: Option<RegexSet>,
}

impl EmailClient {
    /// Create a new email client
    ///
    /// # Arguments
    ///
    /// * `conf` - A Thorium config
    pub async fn new(conf: &Conf) -> Option<Self> {
        // get our email config
        match &conf.thorium.auth.email {
            Some(email_conf) => {
                // build our email credentials
                let creds = Credentials::new(email_conf.addr.clone(), email_conf.password.clone());
                // build our email client
                let client =
                    AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(&email_conf.smtp_server)
                        .expect("Failed to connect to smtp relay")
                        .credentials(creds)
                        .build();
                // get the address to send emails from
                let from = email_conf.addr.parse().expect(&format!(
                    "Failed to parse email address: {}",
                    email_conf.addr
                ));
                // compile our approved email regex if we have any approved emails
                let approved = if !email_conf.approved_emails.is_empty() {
                    // compile our approved emails regex set
                    let approved = RegexSet::new(&email_conf.approved_emails).unwrap();
                    Some(approved)
                } else {
                    None
                };
                // build our email client
                Some(EmailClient {
                    from,
                    client,
                    approved,
                })
            }
            None => None,
        }
    }

    /// Send an email
    pub async fn send<S: Into<String>, B: IntoBody>(
        &self,
        addr: &str,
        subject: S,
        body: B,
    ) -> Result<(), ApiError> {
        // try to parse the email address we are sending email too
        let to = addr.parse().unwrap();
        // build the email to send
        let email = lettre::Message::builder()
            .from(self.from.clone())
            .to(to)
            .subject(subject)
            .header(ContentType::TEXT_PLAIN)
            .body(body)
            .unwrap();
        // send our email
        self.client.send(email).await.unwrap();
        Ok(())
    }
}

/// Shared objects between all requests
pub struct Shared {
    /// The Thorium config f
    pub config: Conf,
    /// A connection pool for redis
    pub redis: Pool<RedisConnectionManager>,
    /// A session for talking to Scylla
    pub scylla: Scylla,
    /// s3 clients for each bucket Thorium uses
    pub s3: S3,
    // The client for Elastic Search
    pub elastic: Elasticsearch,
    /// An email client for verification emails
    pub email: Option<EmailClient>,
    /// A site banner for displaying messages to UI users
    pub banner: String,
}

impl Shared {
    /// Sets up the shared object
    ///
    /// # Arguments
    ///
    /// * `config` - The Thorium config to use
    pub async fn new(config: Conf) -> Self {
        // log the namespace we will be using
        info!(
            config.thorium.tracing.local.level,
            format!("Using namespace {}", config.thorium.namespace)
        );
        // setup redis connection pool
        let redis = retry!(setup::redis(&config), 2, "Redis setup", config);
        // setup scylla session and prepared statements
        let scylla = Scylla::new(&config).await;
        // setup the elastic client
        let elastic = retry!(setup::elastic(&config), 60, "Elastic setup", &config);
        // build an email client if its configured
        let email = EmailClient::new(&config).await;
        // setup s3 clients
        let s3 = S3::new(&config);
        // read banner from local path
        let banner = fs::read_to_string("banner.txt")
            .await
            .unwrap_or("Add your custom Thorium banner here!".to_owned());
        Shared {
            config,
            redis,
            scylla,
            s3,
            elastic,
            email,
            banner,
        }
    }
}

/// All of the global states in Axum
#[derive(Clone)]
pub struct AppState {
    /// The shared objects in Thorium
    pub shared: Arc<Shared>,
}

impl AppState {
    pub fn new(shared: Shared) -> Self {
        AppState {
            shared: Arc::new(shared),
        }
    }
}

impl FromRef<AppState> for Arc<Shared> {
    fn from_ref(state: &AppState) -> Self {
        state.shared.clone()
    }
}
