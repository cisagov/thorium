//! The images related tools for the Thorium MCP server

use rmcp::ErrorData;
use rmcp::handler::server::tool::Extension as RmcpExtension;
use rmcp::model::{CallToolResult, Content};
use rmcp::{tool, tool_router};

use super::ThoriumMCP;

#[tool_router(router = images_router, vis = "pub")]
impl ThoriumMCP {
    /// Get weather information for a city (returns structured data)
    ///
    /// # Arguments
    ///
    /// * `parts` - The request parts required to get a token for this tool
    #[tool(
        name = "list_images",
        description = "List the images or tools in Thorium."
    )]
    pub async fn list_images(
        &self,
        RmcpExtension(parts): RmcpExtension<axum::http::request::Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        // get a thorium client
        let thorium = self.conf.client(&parts).await?;
        // list images in the static group for now
        // TODO: support all groups?
        // TODO: how does pagination work here?
        let mut cursor = thorium.images.list("tree-testing").details().limit(100);
        // get this cursors data
        cursor.next().await?;
        // serialize our list of tools
        let serialized = serde_json::to_value(&cursor.details).unwrap();
        // build our result
        let result = CallToolResult {
            content: Some(vec![Content::json(&cursor.details)?]),
            structured_content: Some(serialized),
            is_error: Some(false),
        };
        Ok(result)
    }
}
