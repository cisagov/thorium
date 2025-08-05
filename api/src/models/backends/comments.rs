//! Wrappers for interacting with comments

use std::collections::HashSet;

use aws_sdk_s3::primitives::ByteStream;
use axum::extract::Multipart;
use tracing::instrument;
use uuid::Uuid;

use super::db;
use crate::models::{CommentForm, CommentResponse, Group, GroupAllowAction, User};
use crate::utils::{ApiError, Shared};
use crate::{bad, can_create_all};

pub trait CommentSupport {
    /// Creates a new comment
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is adding a comment
    /// * `req` - The multipart form containing the new comment
    /// * `shared` - Shared Thorium objects
    #[allow(async_fn_in_trait)]
    async fn create_comment(
        &self,
        user: &User,
        req: Multipart,
        shared: &Shared,
    ) -> Result<CommentResponse, ApiError>;

    /// Deletes a comment
    ///
    /// # Arguments
    ///
    /// * `user` - The user that is deleting the comment
    /// * `groups` - The groups to delete the comment from
    /// * `id` - The id of the comment to delete
    /// * `shared` - Shared Thorium objects
    #[allow(async_fn_in_trait)]
    async fn delete_comment(
        &self,
        user: &User,
        groups: &[String],
        id: &Uuid,
        shared: &Shared,
    ) -> Result<(), ApiError>;

    /// Downloads an attachment from a specific comment
    ///
    /// # Arguments
    ///
    /// * `comment` - The id of the comment to download
    /// * `attachment` - The id of the attachment to download
    /// * `shared` - Shared Thorium objects
    #[allow(async_fn_in_trait)]
    async fn download_attachment(
        &self,
        comment: &Uuid,
        attachment: &Uuid,
        shared: &Shared,
    ) -> Result<ByteStream, ApiError>;
}

/// Helps create a new comment for an object
///
/// # Arguments
///
/// * `user` - The user that is adding new comments
/// * `key` - The key for the object this comment is for
/// * `groups` - The groups to add this comment too if no groups are specified
/// * `req` - The multipart form containing our new comment and any attachments
/// * `form` - The comment form to add our multipart entries too
/// * `shared` - Shared objects in Thorium
#[instrument(
    name = "backends::comments::create_comment_helper",
    skip(user, req, form, shared),
    err(Debug)
)]
pub async fn create_comment_helper(
    user: &User,
    key: &str,
    groups: &HashSet<&str>,
    mut req: Multipart,
    form: &mut CommentForm,
    shared: &Shared,
) -> Result<(), ApiError> {
    // copy our comment id
    let comment_id = form.id;
    // begin crawling over our multipart form upload
    while let Some(field) = req.next_field().await? {
        // try to consume our fields
        if let Some(data_field) = form.add(field).await? {
            // throw an error if the correct content type is not used
            if data_field.content_type().is_none() {
                return bad!("A content type must be set for the data form entry!".to_owned());
            }
            // generate a random uuid for this comment attachment
            let s3_id = Uuid::new_v4();
            // try to get the name for this file
            let file_name = data_field.file_name().map(|name| name.to_owned());
            // build the path to save this attachment at in s3
            let s3_path = format!("{}/{}/{}", key, &comment_id, s3_id);
            // cart and stream this file into s3
            shared.s3.attachments.stream(&s3_path, data_field).await?;
            // add this file name to our form
            // otherwise set the file name as the S3 UUID
            form.attachments
                .insert(file_name.unwrap_or_else(|| s3_id.to_string()), s3_id);
        }
    }
    // provide sane defaults if the user provided no groups; otherwise, check that they are valid
    if form.groups.is_empty() {
        // get the groups we can see this sample in
        form.groups.extend(groups.iter().map(ToString::to_string));
        // make sure we can actually upload files to all the requested groups
        let groups = Group::authorize_check_allow_all(
            user,
            &form.groups,
            GroupAllowAction::Comments,
            shared,
        )
        .await?;
        // make sure we have the roles to upload samples in all of these groups
        can_create_all!(groups, user, shared);
    } else {
        // make sure we can actually upload files to all the requested groups
        let groups = Group::authorize_check_allow_all(
            user,
            &form.groups,
            GroupAllowAction::Comments,
            shared,
        )
        .await?;
        // make sure we have the roles to upload samples in all of these groups
        can_create_all!(groups, user, shared);
    }
    // save the new comment into scylla
    db::files::create_comment(user, key, form, shared).await
}
