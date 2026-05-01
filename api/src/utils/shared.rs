//! Shared objects and methods across all requests
use axum::extract::FromRef;
use bb8_redis::{RedisConnectionManager, bb8::Pool};
use elasticsearch::Elasticsearch;
use lettre::message::header::ContentType;
use lettre::message::{IntoBody, Mailbox};
use lettre::transport::smtp::authentication::Credentials;
use lettre::{AsyncSmtpTransport, AsyncTransport, Tokio1Executor};
use openidconnect::core::{
    CoreAuthDisplay, CoreAuthPrompt, CoreAuthenticationFlow, CoreClient, CoreErrorResponseType,
    CoreGenderClaim, CoreJsonWebKey, CoreJweContentEncryptionAlgorithm, CoreJwsSigningAlgorithm,
    CoreProviderMetadata, CoreRevocableToken, CoreTokenType,
};
use openidconnect::{
    ClientId, ClientSecret, CsrfToken, EmptyAdditionalClaims, EmptyExtraTokenFields,
    EndpointMaybeSet, EndpointNotSet, EndpointSet, IdTokenFields, IssuerUrl, Nonce,
    PkceCodeChallenge, PkceCodeVerifier, RedirectUrl, RevocationErrorResponseType, Scope,
    StandardErrorResponse, StandardTokenIntrospectionResponse, StandardTokenResponse,
};
use regex::RegexSet;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::fs;
use url::Url;

use super::s3::S3;
use crate::conf::OauthProvider;
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

/// A client for any OAuth provider
pub struct OAuthClient {
    /// The name of the provider this client is for
    pub name: String,
    /// The different scopes for this provider
    pub scopes: Vec<String>,
    /// The http client to use to drive this client
    pub http_client: reqwest::Client,
    /// The client to this provider
    pub client: openidconnect::Client<
        EmptyAdditionalClaims,
        CoreAuthDisplay,
        CoreGenderClaim,
        CoreJweContentEncryptionAlgorithm,
        CoreJsonWebKey,
        CoreAuthPrompt,
        StandardErrorResponse<CoreErrorResponseType>,
        StandardTokenResponse<
            IdTokenFields<
                EmptyAdditionalClaims,
                EmptyExtraTokenFields,
                CoreGenderClaim,
                CoreJweContentEncryptionAlgorithm,
                CoreJwsSigningAlgorithm,
            >,
            CoreTokenType,
        >,
        StandardTokenIntrospectionResponse<EmptyExtraTokenFields, CoreTokenType>,
        CoreRevocableToken,
        StandardErrorResponse<RevocationErrorResponseType>,
        EndpointSet,
        EndpointNotSet,
        EndpointNotSet,
        EndpointNotSet,
        EndpointMaybeSet,
        EndpointMaybeSet,
    >,
}

impl OAuthClient {
    /// Create a new client for a provider
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the provider to build a client for
    /// * `redirect_base` - The base sheme + domain to redirect too
    /// * `conf` - The config for this OAuth provider
    pub async fn new(
        name: &str,
        redirect_base: &str,
        conf: &OauthProvider,
    ) -> Result<Self, ApiError> {
        // create a reqwest for this oauth client to use
        let http_client = reqwest::ClientBuilder::new()
            // disable following redirects to prevent SSRF vulnerabilities
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("Client should build");
        // get this providers metadata
        let provider_metadata = CoreProviderMetadata::discover_async(
            IssuerUrl::new(conf.issuer_url.clone()).unwrap(),
            &http_client,
        )
        .await
        .unwrap();
        // build our id and secret values
        let id = ClientId::new(conf.client_id.clone());
        let secret = Some(ClientSecret::new(conf.client_secret.clone()));
        // build our redirect url
        let redirect_url = format!("{redirect_base}/oauth/{name}/callback");
        // build an oauth client from our providers metadata
        let client = CoreClient::from_provider_metadata(provider_metadata, id, secret)
            // set the url to redirect back to after authentication
            .set_redirect_uri(RedirectUrl::new(redirect_url).unwrap());
        // build our oauth client struct
        let wrapper = OAuthClient {
            name: name.to_string(),
            scopes: conf.scopes.clone(),
            http_client,
            client,
        };
        Ok(wrapper)
    }

    /// Generate an auth url for this provider
    pub fn generate_auth_url(&self) -> (Url, CsrfToken, Nonce, PkceCodeVerifier) {
        // generate a PKCE challenge.
        let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();
        // generate the full authorization URL
        let builder = self
            .client
            .authorize_url(
                CoreAuthenticationFlow::AuthorizationCode,
                CsrfToken::new_random,
                Nonce::new_random,
            )
            // Set the PKCE code challenge.
            .set_pkce_challenge(pkce_challenge);
        // add the scopes that this provider needs
        let builder = self.scopes.iter().fold(builder, |builder, scope| {
            builder.add_scope(Scope::new(scope.clone()))
        });
        // get the url for this user to auth with
        let (url, csrf_token, nonce) = builder.url();
        // return all of the relevant info
        (url, csrf_token, nonce, pkce_verifier)
    }
}

/// Setup all of the oauth provider clients mentioned in our config
///
/// # Arguments
///
/// * `config` - The Thorium config to build clients from
async fn setup_oauth_providers(config: &Conf) -> HashMap<String, OAuthClient> {
    // get our oauth config if we have one
    match &config.thorium.auth.oauth {
        Some(oauth_conf) => {
            // preallocate a map for our oauth providers
            let mut provider_map = HashMap::with_capacity(oauth_conf.providers.len());
            // step over the oauth providers and build clients for them
            for (name, provider_conf) in &oauth_conf.providers {
                // build a client for this provider
                let client = OAuthClient::new(name, &oauth_conf.redirect_base, provider_conf)
                    .await
                    .unwrap_or_else(|error| {
                        panic!("Failed to setup Oauth provider {name}: {error:#?}")
                    });
                // add this client to our map
                provider_map.insert(name.to_owned(), client);
            }
            provider_map
        }
        None => HashMap::default(),
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
    /// Client for OAuth based auth
    pub oauth: HashMap<String, OAuthClient>,
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
        // setup our oauth clients
        let oauth = setup_oauth_providers(&config).await;
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
            oauth,
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
