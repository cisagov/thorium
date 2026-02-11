//! The files related tools for the Thorium MCP server

use std::collections::HashMap;
use std::path::PathBuf;

use rmcp::ErrorData;
use rmcp::handler::server::tool::Extension as RmcpExtension;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content, ErrorCode, ResourceContents};
use rmcp::{tool, tool_router};
use schemars::JsonSchema;
use tracing::instrument;
use uuid::Uuid;

use crate::client::ResultsClient;
use crate::models::ResultGetParams;
use crate::not_found_unwrapped;

use super::ThoriumMCP;

/// The params needed to descibe a sample
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct Sha256 {
    /// The sha256 of the sample to get info on
    pub sha256: String,
}

/// The params needed to list all result files for a specific tool
#[derive(Debug, Serialize, Deserialize, JsonSchema, utoipa::ToSchema)]
pub struct SampleListResultsFiles {
    /// The sha256 of the sample to list result files for
    pub sha256: String,
    /// The tool to list result files for
    pub tool: String,
}

/// The params needed to download a specific result file for a sample
#[derive(Debug, Serialize, Deserialize, JsonSchema, utoipa::ToSchema)]
pub struct SampleGetResultsFile {
    /// The sha256 of the sample to get a result file for
    pub sha256: String,
    /// The tool to get a result file for
    pub tool: String,
    /// The relative path for the absolute file to get (from "files" in the result)
    #[schema(value_type = String)]
    pub path: PathBuf,
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

    /// Get all available result files for a tool for specific sample/file
    ///
    /// # Arguments
    ///
    /// * `parameters` - The parameters required for this tool
    /// * `parts` - The request parts required to get a token for this tool
    #[tool(
        name = "list_sample_result_file_paths",
        description = "Get all available result files for a tool for specific sample/file."
    )]
    #[instrument(
        name = "ThoriumMCP::list_sample_result_file_paths",
        skip(self, parts),
        err(Debug)
    )]
    pub async fn list_sample_result_files(
        &self,
        Parameters(SampleListResultsFiles { sha256, tool }): Parameters<SampleListResultsFiles>,
        RmcpExtension(parts): RmcpExtension<axum::http::request::Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        // get a thorium client
        let thorium = self.conf.client(&parts).await?;
        // only get the results for this tool
        let params = ResultGetParams::default().tool(&tool);
        // get the latest results for this sample and tool
        let results = thorium.files.get_results(&sha256, &params).await?;
        // get the latest result id
        let paths = match results.results.get(&tool) {
            Some(output) => {
                // get our first result id
                match output.first() {
                    Some(result) => &result.files,
                    None => {
                        return Err(ErrorData::from(not_found_unwrapped!(format!(
                            "{sha256} doesn't have results for {tool} yet. Maybe it needs to be run?"
                        ))));
                    }
                }
            }
            None => {
                return Err(ErrorData::from(not_found_unwrapped!(format!(
                    "{tool} doesn't exist or doesn't have results for {sha256}"
                ))));
            }
        };
        // serialize our results
        let serialized = serde_json::to_value(&paths).unwrap();
        // build our structured content
        let content = paths
            .iter()
            .map(|path| Content::json(path))
            .collect::<Result<Vec<_>, _>>()?;
        // build our result
        let result = CallToolResult {
            content,
            structured_content: Some(serialized),
            is_error: Some(false),
            meta: None,
        };
        Ok(result)
    }

    /// Get a specific results file from a tool for a specific sample/file by sha256.
    ///
    /// # Arguments
    ///
    /// * `parameters` - The parameters required for this tool
    /// * `parts` - The request parts required to get a token for this tool
    #[tool(
        name = "get_sample_result_file",
        description = "Get a specific results file from a tool for a specific sample/sha256. \
        Result file paths come from the files list in results."
    )]
    #[instrument(
        name = "ThoriumMCP::get_sample_result_file",
        skip(self, parts),
        err(Debug)
    )]
    pub async fn get_sample_result_file(
        &self,
        Parameters(SampleGetResultsFile { sha256, tool, path }): Parameters<SampleGetResultsFile>,
        RmcpExtension(parts): RmcpExtension<axum::http::request::Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        // get a thorium client
        let thorium = self.conf.client(&parts).await?;
        // only get the results for this tool
        let params = ResultGetParams::default().tool(&tool);
        // get the latest results for this sample and tool
        let results = thorium.files.get_results(&sha256, &params).await?;
        // get the latest result id
        let result_id = match results.results.get(&tool) {
            Some(output) => {
                // get our first result id
                match output.first() {
                    Some(result) => result.id,
                    None => {
                        return Err(ErrorData::from(not_found_unwrapped!(format!(
                            "{sha256} doesn't have results for {tool} yet. Maybe it needs to be run?"
                        ))));
                    }
                }
            }
            None => {
                return Err(ErrorData::from(not_found_unwrapped!(format!(
                    "{tool} doesn't exist or doesn't have results for {sha256}"
                ))));
            }
        };
        // build a uri for this resource
        let uri = format!("{sha256}/{tool}/{result_id}/{}", path.display());
        // try to download this result file
        let result = thorium
            .files
            .download_result_file(&sha256, &tool, &result_id, path)
            .await?;
        // turn our result into text
        let text = String::from_utf8_lossy(&result.data);
        // wrap this result file in a resource
        let resource = ResourceContents::text(text, uri);
        // build our result
        let result = CallToolResult {
            content: vec![Content::resource(resource)],
            structured_content: None,
            is_error: Some(false),
            meta: None,
        };
        Ok(result)
    }
}
