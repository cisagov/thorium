//! MCP support for Thorium

use axum::Router;
use axum::http::request::Parts;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::transport::StreamableHttpServerConfig;
use rmcp::transport::streamable_http_server::StreamableHttpService;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::{ErrorData, tool_handler};
use std::net::IpAddr;
use std::str::FromStr;

mod files;
mod images;
mod pipelines;
mod trees;

use crate::Conf;
use crate::client::Thorium;
use crate::utils::AppState;

/// The config info needed for mcp clients
#[derive(Clone, Copy)]
pub struct McpConfig {
    // The ip address to talk to the API at
    pub ip: IpAddr,
    /// The port to ues to talk to our api
    pub port: u16,
}

impl McpConfig {
    /// Get the url to talk to the Thorium API at
    pub fn get_url(&self) -> String {
        // build the url to talk to Thorium at
        format!("http://{}:{}", self.ip, self.port)
    }

    /// Grab our token from this requests parts
    ///
    /// # Arguments
    ///
    /// * `parts` - The parts to grab our token from
    fn grab_token(parts: &Parts) -> Result<&str, ErrorData> {
        // get our authorizaton header if it exists
        match parts.headers.get("Authorization") {
            Some(value) => {
                // decode the header as a str without panicking so a non-UTF8 Authorization
                // header returns a clean INVALID_PARAMS error instead of crashing the request task
                let value_str = value.to_str().map_err(|_| ErrorData {
                    code: rmcp::model::ErrorCode::INVALID_PARAMS,
                    message: "Authorization header is not valid UTF-8".into(),
                    data: None,
                })?;
                // split this on spaces and get our token
                // if there isn't a space then assume they passed just a token
                match value_str.split_once(' ') {
                    Some((_, token)) => Ok(token),
                    None => Ok(value_str),
                }
            }
            // we are missing an authorization header to reject this request
            None => Err(ErrorData {
                code: rmcp::model::ErrorCode::INVALID_PARAMS,
                message: "Missing authorization header".into(),
                data: None,
            }),
        }
    }

    /// Get a Thorium client to use for this MCP session
    ///
    /// # Arguments
    ///
    /// * `parts` - The request parts to get token info from
    pub async fn client(&self, parts: &Parts) -> Result<Thorium, ErrorData> {
        // build the url to talk to Thorium at
        let url = self.get_url();
        // get our authorization token
        let token = Self::grab_token(parts)?;
        // get a thorim client
        let thorium = Thorium::build(&url).token(token).build().await?;
        Ok(thorium)
    }
}

impl From<&Conf> for McpConfig {
    fn from(conf: &Conf) -> McpConfig {
        // get our interface as an ip address
        let ip = IpAddr::from_str(&conf.thorium.interface).unwrap();
        // build our mcp config
        McpConfig {
            ip,
            port: conf.thorium.port,
        }
    }
}

#[derive(Clone)]
pub struct ThoriumMCP {
    conf: McpConfig,
    tool_router: ToolRouter<Self>,
}

#[tool_handler(router = self.tool_router)]
impl rmcp::ServerHandler for ThoriumMCP {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some("A file analysis and data generation platform.".into()),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

impl ThoriumMCP {
    /// Build a new Thorium MCP instance
    ///
    /// # Arguments
    ///
    /// * `mcp_conf` - The mcp config to use
    pub fn new(mcp_conf: McpConfig) -> Self {
        Self {
            conf: mcp_conf,
            tool_router: Self::sample_router()
                + Self::images_router()
                + Self::pipelines_router()
                + Self::tree_router(),
        }
    }
}

/// Mount the mcp service to our api
///
/// # Arguments
///
/// * `router` - The router to add routes too
/// * `conf` - The Thorium config
pub fn mount(router: Router<AppState>, conf: &Conf) -> Router<AppState> {
    // create a new mcp config
    let mcp_conf = McpConfig::from(conf);
    // create a new service
    let service = StreamableHttpService::new(
        move || Ok(ThoriumMCP::new(mcp_conf)),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    // get a new mcp router
    let mcp_router = Router::<AppState>::new().fallback_service(service);
    // nest our router
    router.nest("/mcp", mcp_router)
}

#[cfg(test)]
mod tests {
    use super::McpConfig;
    use axum::http::request::Parts;
    use axum::http::{HeaderValue, Request};

    /// Build request parts carrying the given raw Authorization header bytes
    ///
    /// # Arguments
    ///
    /// * `value` - The raw bytes to set as the Authorization header value
    fn parts_with_auth(value: &[u8]) -> Parts {
        // craft a header value straight from bytes so we can inject non-UTF8 input
        let header = HeaderValue::from_bytes(value).expect("failed to build header value");
        // build a request carrying that header and tear off just its parts
        Request::builder()
            .header("Authorization", header)
            .body(())
            .expect("failed to build request")
            .into_parts()
            .0
    }

    #[test]
    fn grab_token_rejects_non_utf8_authorization() {
        // a non-UTF8 Authorization header is attacker controlled and must not crash the API
        let parts = parts_with_auth(&[0xFF, 0xFE]);
        // grabbing the token must return an error instead of panicking on `to_str().unwrap()`
        assert!(
            McpConfig::grab_token(&parts).is_err(),
            "grab_token accepted a non-UTF8 Authorization header instead of erroring"
        );
    }

    #[test]
    fn grab_token_extracts_bearer_token() {
        // a well formed bearer header must still yield the token after the space
        let parts = parts_with_auth(b"Bearer deadbeefcafe");
        let token = McpConfig::grab_token(&parts).expect("failed to grab a valid token");
        assert_eq!(token, "deadbeefcafe");
    }
}
