//! The trees related tools for the Thorium MCP server

use rmcp::ErrorData;
use rmcp::handler::server::tool::Extension as RmcpExtension;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content};
use rmcp::{tool, tool_router};
use schemars::JsonSchema;
use serde_with::OneOrMany;
use serde_with::formats::PreferMany;
use std::collections::{BTreeMap, BTreeSet};
use tracing::instrument;
use uuid::Uuid;

use crate::models::{TreeOpts, TreeQuery, TreeRelatedQuery};

use super::ThoriumMCP;

/// The params needed to start a new tree
#[serde_with::serde_as]
#[derive(Debug, Serialize, Deserialize, JsonSchema, utoipa::ToSchema)]
pub struct StartTree {
    /// The sha256 for samples to start growing a tree of related data from
    #[serde(default)]
    #[serde_as(as = "OneOrMany<_, PreferMany>")]
    #[schema(value_type = Vec<String>, example = "[\"63b0490d4736e740f26ea9483d55c254abe032845b70ba84ea463ca6582d106f\"]")]
    pub samples: Vec<String>,
    /// The different repo urls (example: ["https://github.com/cisagov/thorium"]) to build this tree from
    #[serde(default)]
    #[serde_as(as = "OneOrMany<_, PreferMany>")]
    #[schema(value_type = Vec<String>, example = "[\"github.com/cisagov/thorium\"]")]
    pub repos: Vec<String>,
    /// The entity ids to build this tree from
    #[serde(default)]
    #[serde_as(as = "OneOrMany<_, PreferMany>")]
    #[schema(value_type = Vec<String>, example = "[\"26acf2d8-e0a4-4909-b9db-39bf7bc098f0\"]")]
    pub entities: Vec<Uuid>,
    /// The different tag filters (example: {"<KEY>": ["<VALUE>"]}) to build this tree from
    #[serde(default)]
    pub tags: Vec<BTreeMap<String, BTreeSet<String>>>,
}

#[tool_router(router = tree_router, vis = "pub")]
impl ThoriumMCP {
    /// Find data related to a sample, repo, tag, or entity by crawling data in thorium in a tree structure.
    ///
    /// # Arguments
    ///
    /// * `parameters` - The parameters required for this tool
    /// * `parts` - The request parts required to get a token for this tool
    #[tool(
        name = "start_tree",
        description = "Find data related to a sample, repo, tag, or entity by crawling data in thorium in a tree structure."
    )]
    #[instrument(name = "ThoriumMCP::start_tree", skip(self, parts), err(Debug))]
    pub async fn start_tree(
        &self,
        Parameters(StartTree {
            samples,
            repos,
            entities,
            tags,
        }): Parameters<StartTree>,
        RmcpExtension(parts): RmcpExtension<axum::http::request::Parts>,
    ) -> Result<CallToolResult, ErrorData> {
        // build the query for starting a new tree
        let query = TreeQuery {
            groups: Vec::default(),
            samples,
            repos,
            entities,
            tags,
            related: TreeRelatedQuery::default(),
        };
        // use default query options
        let opts = TreeOpts::default();
        // get a thorium client
        let thorium = self.conf.client(&parts).await?;
        // grow a tree based on our initial query
        let tree = thorium.trees.start(&opts, &query).await?;
        // serialize our tree
        let serialized = serde_json::to_value(&tree).unwrap();
        // build our result
        let result = CallToolResult {
            content: vec![Content::json(&tree)?],
            structured_content: Some(serialized),
            is_error: Some(false),
            meta: None,
        };
        Ok(result)
    }
}
