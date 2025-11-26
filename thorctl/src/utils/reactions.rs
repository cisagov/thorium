//! Utility functions relating to reactions

use thorium::{Error, Thorium, models::Reaction};
use uuid::Uuid;

/// Search for a particular reaction in every group that a user is in; useful when
/// no specific reaction group is given
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `reaction_id` - The id of the reaction to find
pub async fn find_reaction_no_group(
    thorium: &Thorium,
    reaction_id: &Uuid,
) -> Result<Reaction, Error> {
    // get all groups for the current user
    let groups = super::groups::get_all_groups(thorium).await?;
    // concurrently query for the reaction in every group
    let reaction_search_results = futures::future::join_all(
        groups
            .into_iter()
            .map(|group| async move { thorium.reactions.get(&group, reaction_id.clone()).await }),
    )
    .await;
    // look for a valid reaction among the results
    for res in reaction_search_results {
        if res.is_ok() {
            return res;
        }
    }
    // return a 404 NOT FOUND error if the reaction is not in any of the groups
    Err(Error::Thorium {
        code: http::StatusCode::NOT_FOUND,
        msg: Some(format!(
            "Reaction {reaction_id} not found in any of the user's groups"
        )),
    })
}
