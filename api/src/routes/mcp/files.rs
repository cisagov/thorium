//! The files related tools for the Thorium MCP server

use std::collections::HashMap;

use rmcp::ErrorData;
use rmcp::handler::server::tool::Extension as RmcpExtension;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content};
use rmcp::{tool, tool_router};
use schemars::JsonSchema;
use tracing::instrument;

use crate::client::ResultsClient;
use crate::models::ResultGetParams;

use super::ThoriumMCP;

/// The params needed to descibe a sample
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct Sha256 {
    /// The sha256 of the sample to get info on
    pub sha256: String,
}

#[tool_router(router = sample_router, vis = "pub")]
impl ThoriumMCP {
    /// Get basic info about a specific sample/file by sha256
    ///
    /// # Arguments
    ///
    /// * `parameters` - The parameters required for this tool
    /// * `parts` - The request parts required to get a token for this tool
    #[tool(
        name = "get_sample",
        description = "Get basic info about a specific sample/file by sha256."
    )]
    #[instrument(name = "ThoriumMCP::get_sample", skip(self, parts), err(Debug))]
    pub async fn get_sample(
        &self,
        Parameters(Sha256 { sha256 }): Parameters<Sha256>,
        RmcpExtension(parts): RmcpExtension<axum::http::request::Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        // get a thorium client
        let thorium = self.conf.client(&parts).await?;
        // get this sample
        let sample = thorium.files.get(&sha256).await?;
        // serialize our sample
        let serialized = serde_json::to_value(&sample).unwrap();
        // build our result
        let result = CallToolResult {
            content: vec![Content::json(&sample)?],
            structured_content: Some(serialized),
            is_error: Some(false),
            meta: None,
        };
        Ok(result)
    }

    /// Get tool results for a specific sample/file by sha256.
    ///
    /// # Arguments
    ///
    /// * `parameters` - The parameters required for this tool
    /// * `parts` - The request parts required to get a token for this tool
    #[tool(
        name = "get_sample_results",
        description = "Get tool results for a specific sample/file by sha256."
    )]
    #[instrument(name = "ThoriumMCP::get_sample_results", skip(self, parts), err(Debug))]
    pub async fn get_sample_results(
        &self,
        Parameters(Sha256 { sha256 }): Parameters<Sha256>,
        RmcpExtension(parts): RmcpExtension<axum::http::request::Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        // get a thorium client
        let thorium = self.conf.client(&parts).await?;
        // build our results get params
        let params = ResultGetParams::default();
        // get this sample's results
        let sample_results = thorium.files.get_results(&sha256, &params).await?;
        // build a nicer map of results for ai
        let mut nicer = HashMap::with_capacity(20);
        for (name, mut result) in sample_results.results {
            nicer.insert(name, result.remove(0).result);
        }
        // serialize our results
        let serialized = serde_json::to_value(&nicer).unwrap();
        // build our result
        let result = CallToolResult {
            content: vec![Content::json(&nicer)?],
            structured_content: Some(serialized),
            is_error: Some(false),
            meta: None,
        };
        Ok(result)
    }
}
