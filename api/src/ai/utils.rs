//! Utiltities for AI in Thorium

use rmcp::model::{
    CallToolRequestParam, CallToolResult, ClientCapabilities, ClientInfo, Implementation,
    InitializeRequestParam, ProtocolVersion,
};
use rmcp::service::RunningService;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::{RoleClient, ServiceExt};
use uuid::Uuid;

use crate::{CtlConf, Error};

/// Setup an mcp client
///
/// # Arguments
///
/// * `conf` - A Thorctl config
pub async fn setup_mcp(
    conf: &CtlConf,
) -> Result<RunningService<RoleClient, InitializeRequestParam>, Error> {
    // build the url to thorium's mcp routes
    let mcp_uri = format!("{}/api/mcp", conf.keys.api);
    // make sure we have a token
    let token = match &conf.keys.token {
        Some(token) => token,
        // we can't use password auth with mcp so throw an error
        None => return Err(Error::new("Please run thorctl login first!")),
    };
    // build the config to use with this transport
    let mut config = StreamableHttpClientTransportConfig::with_uri(mcp_uri).auth_header(token);
    // make our mcp client stateless
    config.allow_stateless = true;
    // setup our transport
    let transport = StreamableHttpClientTransport::from_config(config);
    // build our client
    let client_info = ClientInfo {
        protocol_version: ProtocolVersion::default(),
        capabilities: ClientCapabilities::default(),
        client_info: Implementation {
            name: "Thorium".to_owned(),
            title: Some("Thorium".to_owned()),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            icons: None,
            website_url: Some(conf.keys.api.clone()),
        },
    };
    // build our client
    let mcp_client = client_info.serve(transport).await.unwrap();
    Ok(mcp_client)
}

/// Help ThorChat call some mcp based tool
///
/// # Arguments
///
/// * `mcp` - The mcp server to talk too
/// * `id` - The id of the tool to call
/// * `params` - The params to use when calling this tool
pub(super) async fn call_tool_helper(
    mcp: &RunningService<RoleClient, InitializeRequestParam>,
    id: Uuid,
    params: CallToolRequestParam,
) -> Result<(Uuid, String, CallToolResult), Error> {
    // get this tools name
    let name = params.name.to_string();
    // call this mcp tool
    let call_resp = mcp.call_tool(params).await?;
    Ok((id, name, call_resp))
}
