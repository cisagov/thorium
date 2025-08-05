//! Test files routes

use data_encoding::HEXLOWER;
use md5::Md5;
use sha1::{Digest, Sha1};
use sha2::Sha256;
use std::collections::HashSet;
use thorium::client::ResultsClient;
use thorium::test_utilities::{self, generators};
use thorium::utils::s3::S3;
use thorium::{
    contains, fail, has_tag, is, is_desc, is_empty, is_in, is_not, is_not_in, no_tag, starts_with,
    vec_in_vec,
};
use uuid::Uuid;

use thorium::models::{
    Buffer, CommentRequest, DeleteCommentParams, FileDeleteOpts, FileDownloadOpts, FileListOpts,
    GroupUpdate, GroupUsersUpdate, ImageVersion, OnDiskFile, OriginRequest, OutputDisplayType,
    OutputRequest, ResultGetParams, SampleRequest, SubmissionUpdate, TagDeleteRequest, TagRequest,
};

#[tokio::test]
async fn create() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new("/bin/sh", vec![group])
        .description("test file")
        .tag("test", "file")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    client.files.create(file_req).await?;
    Ok(())
}

#[tokio::test]
async fn download() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("EvilCorn"), vec![group])
        .description("test file")
        .tag("corn", "yes")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    client.files.create(file_req).await?;
    // build the options for downloading this file
    let mut opts = FileDownloadOpts::default().uncart();
    // download this file
    client
        .files
        .download(
            "afe19e37584cf1d9983889200ca3a8da7957fe6524e91068e9708f07a2f2e79d",
            "UNCARTED_MAL",
            &mut opts,
        )
        .await?;
    // read in our uncarted file
    let data = tokio::fs::read("UNCARTED_MAL").await?;
    // delete the uncarted malware file
    tokio::fs::remove_file("UNCARTED_MAL").await?;
    // make sure that our uncarted file matches
    is!(data, b"EvilCorn");
    Ok(())
}

#[tokio::test]
async fn get() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("leedleleedle"), vec![group])
        .description("wumbo")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let resp = client.files.create(file_req.clone()).await?;
    // get this file and make sure it matches
    let sample = client.files.get(&resp.sha256).await?;
    is!(file_req, sample);
    Ok(())
}

#[tokio::test]
async fn update() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("leedleleedle"), vec![group])
        .description("wumbo")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let resp = client.files.create(file_req.clone()).await?;
    // build the update for this file
    let update = SubmissionUpdate::new(resp.id)
        .name("hash-slinging slasher")
        .origin(OriginRequest::downloaded(
            "https://test.com",
            Some("test".to_string()),
        ));
    // update this file
    let update_resp = client.files.update(&resp.sha256, &update).await?;
    is!(update_resp.status().as_u16(), 204);
    // make sure this tag was updated
    let sample = client.files.get(&resp.sha256).await?;
    is!(sample, update);
    Ok(())
}

#[tokio::test]
async fn update_add_groups() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group_names: Vec<String> = generators::groups(3, &client)
        .await?
        .into_iter()
        .map(|group| group.name)
        .collect();
    // build a sample request
    let file_req = SampleRequest::new_buffer(
        Buffer::new("leedleleedle_add_groups"),
        vec![group_names.get(2).unwrap().clone()],
    )
    .description("wumbo")
    .origin(OriginRequest::downloaded(
        "https://google.com",
        Some("google".to_string()),
    ));
    // upload this file
    let resp = client.files.create(file_req.clone()).await?;

    // build the update for this file
    let mut update = SubmissionUpdate::new(resp.id);
    // add groups to be removed to update request
    let added_group1 = group_names.get(0).unwrap();
    let added_group2 = group_names.get(1).unwrap();
    update.add_groups.push(added_group1.clone());
    update.add_groups.push(added_group2.clone());
    // update file and add groups
    let update_resp = client.files.update(&resp.sha256, &update).await?;
    is!(update_resp.status().as_u16(), 204);
    // make sure the groups were added
    let sample = client.files.get(&resp.sha256).await?;
    is_in!(sample.groups(), added_group1.as_str());
    is_in!(sample.groups(), added_group2.as_str());

    // Try to add already added group
    let mut update = SubmissionUpdate::new(resp.id);
    update
        .add_groups
        .push(group_names.get(2).unwrap().to_string());
    // update file and attempt to add group
    let result = client.files.update(&resp.sha256, &update).await;
    // Make sure there is an error
    fail!(result, 400);
    Ok(())
}

#[tokio::test]
async fn update_remove_groups() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group_names: Vec<String> = generators::groups(3, &client)
        .await?
        .into_iter()
        .map(|group| group.name)
        .collect();
    // build a sample request
    let file_req = SampleRequest::new_buffer(
        Buffer::new("leedleleedle_remove_groups"),
        group_names.clone(),
    )
    .description("wumbo")
    .origin(OriginRequest::downloaded(
        "https://google.com",
        Some("google".to_string()),
    ));
    // upload this file
    let resp = client.files.create(file_req.clone()).await?;
    // build the update for this file
    let mut update = SubmissionUpdate::new(resp.id);

    // add groups to be removed to update request
    let removed_group1 = group_names.get(0).unwrap();
    let removed_group2 = group_names.get(1).unwrap();
    update.remove_groups.push(removed_group1.clone());
    update.remove_groups.push(removed_group2.clone());
    // update file and remove groups
    let update_resp = client.files.update(&resp.sha256, &update).await?;
    is!(update_resp.status().as_u16(), 204);
    // make sure the groups were removed
    let sample = client.files.get(&resp.sha256).await?;
    is_not_in!(sample.groups(), removed_group1.as_str());
    is_not_in!(sample.groups(), removed_group2.as_str());

    // Try to remove last group
    let mut update = SubmissionUpdate::new(resp.id);
    update
        .remove_groups
        .push(group_names.get(2).unwrap().to_string());
    // Update file and attempt to remove last group
    let result = client.files.update(&resp.sha256, &update).await;
    // Make sure there is an error
    fail!(result, 400);

    Ok(())
}

#[tokio::test]
async fn tag() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("leedleleedle"), vec![&group])
        .description("wumbo")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req.clone()).await?;
    // build the tags to add to this sample
    let tag_req = TagRequest::default()
        .group(&group)
        .add_values("plants", vec!["corn", "apples"])
        .add("healthy", "yes");
    // add this tag
    client.files.tag(&hashes.sha256, &tag_req).await?;
    // make sure this tag was added
    let sample = client.files.get(&hashes.sha256).await?;
    is!(sample, tag_req);
    Ok(())
}

#[tokio::test]
async fn delete_tag() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create three groups
    let groups: Vec<String> = generators::groups(3, &client)
        .await?
        .into_iter()
        .map(|group_req| group_req.name)
        .collect();
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("mayonnaise"), groups.clone())
        .description("wumbo")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req.clone()).await?;
    // build the tags to add to this sample
    let tag_req = TagRequest::default()
        .groups(groups.clone())
        .add_values("plants", vec!["corn", "apples"])
        .add("healthy", "yes");
    // add this tag to both groups
    client.files.tag(&hashes.sha256, &tag_req).await?;
    // delete the tags from the first group
    let tag_req = TagDeleteRequest::default()
        .group(groups[0].clone())
        .add_values("plants", vec!["corn", "apples"])
        .add("healthy", "yes");
    client.files.delete_tags(&hashes.sha256, &tag_req).await?;
    // retrieve the sample
    let sample = client.files.get(&hashes.sha256).await?;
    // make sure the tags were deleted
    no_tag!(&sample.tags, "plants", "corn", &groups[0]);
    no_tag!(&sample.tags, "plants", "apples", &groups[0]);
    no_tag!(&sample.tags, "healthy", "yes", &groups[0]);
    // make sure the tags for the second group are still intact
    has_tag!(&sample.tags, "plants", "corn", &groups[1]);
    has_tag!(&sample.tags, "plants", "apples", &groups[1]);
    has_tag!(&sample.tags, "healthy", "yes", &groups[1]);
    // delete only one tag from the second group
    let tag_req = TagDeleteRequest::default()
        .group(groups[1].clone())
        .add("healthy", "yes");
    client.files.delete_tags(&hashes.sha256, &tag_req).await?;
    // retrieve the sample
    let sample = client.files.get(&hashes.sha256).await?;
    // make sure one tag was deleted but the others are intact
    no_tag!(&sample.tags, "healthy", "yes", &groups[1]);
    has_tag!(&sample.tags, "plants", "corn", &groups[1]);
    has_tag!(&sample.tags, "plants", "apples", &groups[1]);
    // delete all of the rest of the tags, not specifying groups to delete from all
    let tag_req = TagDeleteRequest::default()
        .add_values("plants", vec!["corn", "apples"])
        .add("healthy", "yes");
    client.files.delete_tags(&hashes.sha256, &tag_req).await?;
    let sample = client.files.get(&hashes.sha256).await?;
    no_tag!(&sample.tags, "plants");
    no_tag!(&sample.tags, "healthy");
    Ok(())
}

/// Builds the sha256, sha1, and MD5 for a buffer
///
/// # Arguments
///
/// * `reqs` - The sample requests to hash
///
/// # Panics
///
/// Panics if the request does not contain a buffer
#[must_use]
pub fn get_hashes(
    reqs: &Vec<SampleRequest>,
) -> (HashSet<String>, HashSet<String>, HashSet<String>) {
    // create our hashsets
    let mut sha1s = HashSet::with_capacity(reqs.len());
    let mut sha256s = HashSet::with_capacity(reqs.len());
    let mut md5s = HashSet::with_capacity(reqs.len());
    for req in reqs {
        // build hashers for each type
        let mut sha1 = Sha1::new();
        let mut sha256 = Sha256::new();
        let mut md5 = Md5::new();
        // update our hashers with our newly read data
        sha1.update(&req.data.as_ref().unwrap().data);
        sha256.update(&req.data.as_ref().unwrap().data);
        md5.update(&req.data.as_ref().unwrap().data);
        // build a digest for this
        sha1s.insert(HEXLOWER.encode(&sha1.finalize()));
        sha256s.insert(HEXLOWER.encode(&sha256.finalize()));
        md5s.insert(HEXLOWER.encode(&md5.finalize()));
    }
    (sha1s, sha256s, md5s)
}

#[tokio::test]
async fn list() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create 20 random files in this group
    let reqs = generators::samples(&group, 20, &client).await?;
    // build the options for the group we just created
    let opts = FileListOpts::default().groups(vec![&group]);
    // list the 20 files we just created
    let cursor = client.files.list(&opts).await?;
    // build a list of hashes for the files we created
    let (_, sha256s, _) = get_hashes(&reqs);
    // make sure we listed our sha256s
    for item in &cursor.data {
        is_in!(sha256s, item.sha256);
    }
    // make sure our list is in descending order
    is_desc!(cursor.data);
    Ok(())
}

#[tokio::test]
async fn list_details() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create 20 random files in this group
    let reqs = generators::samples(&group, 20, &client).await?;
    // build the options to limit to just the group we just created
    let opts = FileListOpts::default().groups(vec![&group]);
    // list the 20 files we just created
    let cursor = client.files.list_details(&opts).await?;
    // build a list of hashes for the files we created
    let (sha1s, sha256s, md5s) = get_hashes(&reqs);
    // make sure our hashes were correctly set
    for item in &cursor.data {
        is_in!(sha1s, item.sha1);
        is_in!(sha256s, item.sha256);
        is_in!(md5s, item.md5);
        // make sure our submission lists for each sample is in descending order
        is_desc!(item.submissions);
    }
    // make sure our file reqs are in the samples returned
    vec_in_vec!(reqs, cursor.data);
    Ok(())
}

#[tokio::test]
async fn list_tag() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create 20 random files in this group
    let (key, value, reqs) = generators::samples_tagged(&group, 20, &client).await?;
    // build a opts for the group we just created
    let opts = FileListOpts::default().groups(vec![&group]).tag(key, value);
    // list the 20 files we just created
    let cursor = client.files.list(&opts).await?;
    // build a list of hashes for the files we created
    let (_, sha256s, _) = get_hashes(&reqs);
    // make sure we listed our sha256s
    for item in cursor.data {
        is_in!(sha256s, item.sha256);
    }
    Ok(())
}

#[tokio::test]
async fn list_tag_details() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create 20 random files in this group
    let (key, value, reqs) = generators::samples_tagged(&group, 20, &client).await?;
    // build the options for the group we just created
    let opts = FileListOpts::default().groups(vec![&group]).tag(key, value);
    // list the 20 files we just created
    let cursor = client.files.list_details(&opts).await?;
    // build a list of hashes for the files we created
    let (sha1s, sha256s, md5s) = get_hashes(&reqs);
    // make sure our hashes were correctly set
    for item in &cursor.data {
        is_in!(sha1s, item.sha1);
        is_in!(sha256s, item.sha256);
        is_in!(md5s, item.md5);
    }
    // make sure our file reqs are in the samples returned
    vec_in_vec!(reqs, cursor.data);
    Ok(())
}

#[tokio::test]
async fn list_tag_case_insensitive() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    let key = "KEY_CASE_TEST";
    let value = "VALUE_CASE_TEST";
    let key_lower = key.to_lowercase();
    let value_lower = value.to_lowercase();
    // create 20 files with the upper key and 20 with the lower key
    let reqs = generators::samples_with_tag(&group, 20, key, value, &client).await?;
    let reqs_lower =
        generators::samples_with_tag(&group, 20, &key_lower, &value_lower, &client).await?;
    // build the a list of hashes for the files we created
    let (_, mut sha256s, _) = get_hashes(&reqs);
    let (_, sha256s_lower, _) = get_hashes(&reqs_lower);
    // first list just the regular tags as a sanity check
    let opts = FileListOpts::default()
        .groups(vec![&group])
        .tag(key, value)
        .page_size(50)
        .limit(50);
    let cursor = client.files.list(&opts).await?;
    // make sure we listed the right ones
    for item in &cursor.data {
        contains!(sha256s, &item.sha256);
    }
    // make sure we didn't get the lowercase ones
    for item in cursor.data {
        is_not_in!(sha256s_lower, item.sha256);
    }
    // do a case insensitive search
    let opts = FileListOpts::default()
        .groups(vec![&group])
        .tag(key.to_lowercase(), value.to_lowercase())
        .tags_case_insensitive()
        .page_size(50)
        .limit(50);
    let cursor = client.files.list(&opts).await?;
    sha256s.extend(sha256s_lower);
    // make sure we listed our all of them
    for item in cursor.data {
        is_in!(sha256s, item.sha256);
    }
    Ok(())
}

#[tokio::test]
async fn comment() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("not_a_comment"), vec![group])
        .description("also not a comment")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req.clone()).await?;
    // build a comment request for this file
    let comment_req = CommentRequest::new(&hashes.sha256, "I am a comment")
        .file(OnDiskFile::new("/bin/sh").trim_prefix("/bin/"))
        .buffer(Buffer::new("I am an attachment"));
    // comment on this file
    let resp = client.files.comment(comment_req.clone()).await?;
    // get the file we just commented on
    let sample = client.files.get(&hashes.sha256).await?;
    // make sure this comment was added
    is!(sample.comments.iter().any(|com| com.id == resp.id), true);
    is!(sample, comment_req);
    Ok(())
}

#[tokio::test]
async fn comment_download_attachment() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group: String = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("not_a_comment2"), vec![group])
        .description("also not a comment")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req.clone()).await?;
    // build a comment request for this file
    let comment_req = CommentRequest::new(&hashes.sha256, "I am a comment")
        .buffer(Buffer::new("I am an attachment"));
    // comment on this file
    client.files.comment(comment_req.clone()).await?;
    // get the file we just commented on
    let sample = client.files.get(&hashes.sha256).await?;
    // make sure we have a single comment
    is!(sample.comments.len(), 1);
    // get this comments info
    let id = &sample.comments[0].id;
    let (_, attach) = &sample.comments[0].attachments.iter().next().unwrap();
    // download this files attachment and make sure it matches
    let attachment = client
        .files
        .download_attachment(&hashes.sha256, id, attach)
        .await?;
    is!(attachment.data, "I am an attachment".as_bytes());
    Ok(())
}

#[tokio::test]
async fn delete_comment() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let mut groups: Vec<String> = generators::groups(2, &client)
        .await?
        .into_iter()
        .map(|group| group.name)
        .collect();
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("not_a_comment3"), groups.clone())
        .description("also not a comment3")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req.clone()).await?;
    // build a comment request for this file
    let comment_req = CommentRequest::new(&hashes.sha256, "I am a comment")
        .file(OnDiskFile::new("/bin/sh").trim_prefix("/bin/"))
        .buffer(Buffer::new("I am an attachment"));
    // comment on this file
    let resp = client.files.comment(comment_req.clone()).await?;
    // delete the comment from all groups as the admin user;
    // providing no groups as params will delete from all groups
    client
        .files
        .delete_comment(&hashes.sha256, &resp.id, &DeleteCommentParams::default())
        .await?;
    // ensure the comment was deleted
    let sample = client.files.get(&hashes.sha256).await?;
    is_empty!(sample.comments);
    // create a user
    let user_client = generators::client(&client).await?;
    let username = user_client.users.info().await?.username;
    // add the user to both groups
    let group_update =
        GroupUpdate::default().users(GroupUsersUpdate::default().direct_add(username.clone()));
    futures::future::try_join_all(
        groups
            .iter()
            .map(|group| client.groups.update(group, &group_update)),
    )
    .await?;
    // add a comment as the user
    let comment_req = CommentRequest::new(&hashes.sha256, "I am a second comment")
        .file(OnDiskFile::new("/bin/sh").trim_prefix("/bin/"))
        .buffer(Buffer::new("I am a second attachment"));
    let resp = user_client.files.comment(comment_req.clone()).await?;
    // delete the comment from the first group and ensure the comment is still visible in the other
    client
        .files
        .delete_comment(
            &hashes.sha256,
            &resp.id,
            &DeleteCommentParams::default().group(groups.pop().unwrap()),
        )
        .await?;
    let sample = client.files.get(&hashes.sha256).await?;
    is!(sample.comments.first().unwrap().groups, groups);
    // delete the comment from the last group and ensure it's deleted completely
    client
        .files
        .delete_comment(
            &hashes.sha256,
            &resp.id,
            &DeleteCommentParams::default().group(groups.pop().unwrap()),
        )
        .await?;
    let sample = client.files.get(&hashes.sha256).await?;
    is_empty!(sample.comments);
    Ok(())
}

/// Test cases where comment deletion should fail
#[tokio::test]
async fn delete_comment_fail() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("not_a_comment4"), vec![group.clone()])
        .description("also not a comment4")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req.clone()).await?;
    // build a comment request for this file
    let comment_req = CommentRequest::new(&hashes.sha256, "I am a comment")
        .file(OnDiskFile::new("/bin/sh").trim_prefix("/bin/"))
        .buffer(Buffer::new("I am an attachment"));
    // comment on this file
    let resp = client.files.comment(comment_req.clone()).await?;
    // try to delete a nonexistent comment
    let del_result = client
        .files
        .delete_comment(
            &hashes.sha256,
            &Uuid::new_v4(),
            &DeleteCommentParams::default(),
        )
        .await;
    // check for a "NOT FOUND" error
    fail!(del_result, 404);
    // try to delete the comment from a group it's not in
    let other_group = generators::groups(1, &client).await?.remove(0).name;
    let del_result = client
        .files
        .delete_comment(
            &hashes.sha256,
            &resp.id,
            &DeleteCommentParams::default().groups(vec![group.clone(), other_group.clone()]),
        )
        .await;
    // check for a "BAD REQUEST" error
    fail!(del_result, 400);
    // delete the comment from all groups as a user who doesn't have access to the file/comment
    let user_client = generators::client(&client).await?;
    let username = user_client.users.info().await?.username;
    let del_result = user_client
        .files
        .delete_comment(&hashes.sha256, &resp.id, &DeleteCommentParams::default())
        .await;
    // check for a "NOT FOUND" error
    fail!(del_result, 404);
    // add the user to the group
    let group_update =
        GroupUpdate::default().users(GroupUsersUpdate::default().direct_add(username.clone()));
    client.groups.update(&group, &group_update).await?;
    // now try to delete the comment even though the user is not a submitter
    let del_result = user_client
        .files
        .delete_comment(&hashes.sha256, &resp.id, &DeleteCommentParams::default())
        .await;
    // Check for an "UNAUTHORIZED" error
    fail!(del_result, 401);
    Ok(())
}

/// Tests that attachments are pruned when comments are deleted from all groups
#[tokio::test]
async fn comment_attachment_prune() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create two groups
    let groups: Vec<String> = generators::groups(2, &client)
        .await?
        .into_iter()
        .map(|group_req| group_req.name)
        .collect();
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("not_a_comment5"), groups.clone())
        .description("also not a comment5")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file to both groups
    let hashes = client.files.create(file_req.clone()).await?;
    // build a comment request for this file
    let comment_req = CommentRequest::new(&hashes.sha256, "I am a comment")
        .buffer(Buffer::new("I am an attachment"));
    // comment on this file in both groups
    let comment_id = client.files.comment(comment_req.clone()).await?.id;
    // get the sample we just uploaded
    let sample = client.files.get(&hashes.sha256).await?;
    // get the attachment S3 id from the sample
    // from the sample, get the S3 id of the comment attachment before deletion
    let attachment_id = sample.comments[0].attachments.values().next().unwrap();
    // delete the comment from both groups simultaneously;
    // hopefully testing for race condition between two partial deletes
    let delete_comment_params: Vec<DeleteCommentParams> = groups
        .into_iter()
        .map(|group| DeleteCommentParams::default().group(group))
        .collect();
    futures::future::try_join_all(delete_comment_params.iter().map(|params| {
        client
            .files
            .delete_comment(&hashes.sha256, &comment_id, params)
    }))
    .await?;
    // try to retrieve the attachment
    let attachment_path = format!("{}/{}/{}", &hashes.sha256, &comment_id, attachment_id);
    let s3 = S3::new(&test_utilities::config());
    let attachment_resp = s3.attachments.download(&attachment_path).await;
    // ensure the attachment has been deleted
    match attachment_resp {
        Ok(_) => return Err(thorium::Error::new("Comment attachment was not deleted")),
        Err(err) => {
            // Check for correct code and message
            is!(err.code, 400);
            let msg = err
                .msg
                .expect("S3 get_object error message is empty but it shouldn't be");
            starts_with!(msg, "Failed to get object from s3");
        }
    }
    Ok(())
}

#[tokio::test]
async fn create_result() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new("Cargo.toml", vec![group])
        .description("test file")
        .tag("test", "file")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req).await?;
    // create some results for this file
    let output_req = OutputRequest::new(
        hashes.sha256,
        "TestTool",
        "I am a test result",
        OutputDisplayType::String,
    )
    .tool_version(ImageVersion::Custom("TestVersion1.0".to_string()));
    // send this result to the API
    client.files.create_result(output_req).await?;
    Ok(())
}

#[tokio::test]
async fn create_result_files() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new("Cargo.toml", vec![group])
        .description("test file")
        .tag("test", "file")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req).await?;
    // create some results for this file
    let output_req = OutputRequest::new(
        hashes.sha256.clone(),
        "TestTool",
        "I am a test result",
        OutputDisplayType::String,
    )
    .buffer(Buffer::new("TestBuff").name("buff.txt"))
    .buffer(Buffer::new("TestNest").name("nested/buff.txt"))
    .tool_version(ImageVersion::SemVer(
        semver::Version::parse("1.0.0").unwrap(),
    ));
    // send this result to the API
    client.files.create_result(output_req.clone()).await?;
    // get this result and make sure it matches
    let params = ResultGetParams::default();
    let output = client.files.get_results(&hashes.sha256, &params).await?;
    // make sure our output requests matches
    is!(output, output_req);
    Ok(())
}

#[tokio::test]
async fn get_result() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("GetResult"), vec![group])
        .description("test file")
        .tag("test", "file")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req).await?;
    // create some results for this file
    let output_req = OutputRequest::new(
        hashes.sha256.clone(),
        "TestTool",
        "I am a test result",
        OutputDisplayType::String,
    )
    .tool_version(ImageVersion::SemVer(
        semver::Version::parse("1.2.0").unwrap(),
    ));
    // send this result to the API
    client.files.create_result(output_req.clone()).await?;
    // get this result and make sure it matches
    let params = ResultGetParams::default();
    let output = client.files.get_results(&hashes.sha256, &params).await?;
    // make sure our output requests matches
    is!(output, output_req);
    Ok(())
}

#[tokio::test]
async fn create_hidden_result() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new("Cargo.toml", vec![group])
        .description("test file")
        .tag("test", "file")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req).await?;
    // create some results for this file
    let output_req = OutputRequest::new(
        hashes.sha256.clone(),
        "TestTool",
        "I am a test result",
        OutputDisplayType::Hidden,
    )
    .tool_version(ImageVersion::SemVer(
        semver::Version::parse("1.3.0-rc").unwrap(),
    ));
    // send this result to the API
    client.files.create_result(output_req).await?;
    Ok(())
}

#[tokio::test]
async fn get_hidden_result() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("GetHiddenResult"), vec![group])
        .description("test file")
        .tag("test", "file")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req).await?;
    // create some results for this file
    let output_req = OutputRequest::new(
        hashes.sha256.clone(),
        "TestTool",
        "I am a test result",
        OutputDisplayType::Hidden,
    )
    .tool_version(ImageVersion::SemVer(
        semver::Version::parse("1.4.0-alpha").unwrap(),
    ));
    // send this result to the API
    client.files.create_result(output_req.clone()).await?;
    // get this result and make sure it matches
    let params = ResultGetParams::default().hidden();
    let output = client.files.get_results(&hashes.sha256, &params).await?;
    // make sure our output requests matches
    is!(output, output_req);
    Ok(())
}

#[tokio::test]
async fn get_filter_hidden_result() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("GetFilterHiddenResult"), vec![group])
        .description("test file")
        .tag("test", "file")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req).await?;
    // create some results for this file
    let output_req = OutputRequest::new(
        hashes.sha256.clone(),
        "HiddenTool",
        "I am a test result",
        OutputDisplayType::Hidden,
    )
    .tool_version(ImageVersion::SemVer(
        semver::Version::parse("1.5.0").unwrap(),
    ));
    // send this result to the API
    client.files.create_result(output_req.clone()).await?;
    // get this files results and make sure we don't get it
    let params = ResultGetParams::default();
    let output = client.files.get_results(&hashes.sha256, &params).await?;
    // make sure our results don't contain our hidden result
    is!(output.results.contains_key("HiddenTool"), false);
    Ok(())
}

#[tokio::test]
async fn get_specific_tool() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("GetSpecificTool"), vec![group])
        .description("test file")
        .tag("test", "file")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req).await?;
    // create some results for this file
    let output_req = OutputRequest::new(
        hashes.sha256.clone(),
        "TestTool",
        "I am a test result",
        OutputDisplayType::String,
    );
    // send this result to the API
    client.files.create_result(output_req.clone()).await?;
    // create some results for this file
    let wrong_req = OutputRequest::new(
        hashes.sha256.clone(),
        "WrongTool",
        "I am the wrong test result",
        OutputDisplayType::String,
    );
    // send this result to the API
    client.files.create_result(wrong_req.clone()).await?;
    // get this result and make sure it matches
    let params = ResultGetParams::default().tool("TestTool");
    let output = client.files.get_results(&hashes.sha256, &params).await?;
    // make sure our output requests matches
    is!(output, output_req);
    // make sure we didn't get results for the wrong tool
    is_not!(output, wrong_req);
    Ok(())
}

#[tokio::test]
async fn get_specific_group() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let groups = generators::groups(2, &client)
        .await?
        .into_iter()
        .map(|group| group.name)
        .collect::<Vec<String>>();
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("GetSpecificGroup"), groups.clone())
        .description("test file")
        .tag("test", "file")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req).await?;
    // create some results for this file
    let output_req = OutputRequest::new(
        hashes.sha256.clone(),
        "TestTool",
        "I am a test result",
        OutputDisplayType::String,
    )
    .group(&groups[0]);
    // send this result to the API
    client.files.create_result(output_req.clone()).await?;
    // create some results for this file
    let wrong_req = OutputRequest::new(
        hashes.sha256.clone(),
        "TestTool",
        "I am the wrong test result",
        OutputDisplayType::String,
    )
    .group(&groups[1]);
    // send this result to the API
    client.files.create_result(wrong_req.clone()).await?;
    // get this result and make sure it matches
    let params = ResultGetParams::default().group(&groups[0]);
    let output = client.files.get_results(&hashes.sha256, &params).await?;
    // make sure our output requests matches
    is!(output, output_req);
    // make sure we didn't get results for the wrong group
    is_not!(output, wrong_req);
    Ok(())
}

#[tokio::test]
async fn create_files() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new("/bin/sh", vec![group])
        .description("test file")
        .tag("test", "file")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req).await?;
    // create some results for this file
    let output_req = OutputRequest::new(
        hashes.sha256.clone(),
        "TestTool",
        "I am a test result",
        OutputDisplayType::String,
    )
    .file(OnDiskFile::new("/bin/bash"))
    .file(OnDiskFile::new("/bin/ls"));
    // send this result to the API
    client.files.create_result(output_req).await?;
    Ok(())
}

#[tokio::test]
async fn get_result_files() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("GetResultFiles"), vec![group])
        .description("test file")
        .tag("test", "file")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let hashes = client.files.create(file_req).await?;
    // create some results for this file
    let output_req = OutputRequest::new(
        hashes.sha256.clone(),
        "TestTool",
        "I am a test result",
        OutputDisplayType::String,
    )
    .buffers(vec![Buffer::new("1234").name("1.txt"), Buffer::new("5678")]);
    // send this result to the API
    let result_id = client.files.create_result(output_req.clone()).await?;
    // get this result and make sure it matches
    let params = ResultGetParams::default();
    let output = client.files.get_results(&hashes.sha256, &params).await?;
    // make sure our output requests matches
    is!(output, output_req);
    // download our results files
    let downloaded = client
        .files
        .download_result_file(&hashes.sha256, "TestTool", &result_id.id, "1.txt")
        .await?;
    is!(downloaded.data, "1234".as_bytes());
    Ok(())
}

#[tokio::test]
async fn delete() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("masamune"), vec![group.clone()])
        .description("ultima")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let resp = client.files.create(file_req.clone()).await?;
    // delete the file
    client
        .files
        .delete(&resp.sha256, &resp.id, &FileDeleteOpts::default())
        .await?;
    // check that the file was deleted
    let result = client.files.get(&resp.sha256).await;
    fail!(result, 404);
    Ok(())
}

/// Test if submitter tag is deleted after submission deletion
#[tokio::test]
async fn delete_from_group() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group_reqs = generators::groups(4, &client).await?;
    let mut groups: Vec<String> = group_reqs.into_iter().map(|req| req.name).collect();
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("climhazzard"), groups.clone())
        .description("braver")
        .origin(OriginRequest::downloaded(
            "https://google.com",
            Some("google".to_string()),
        ));
    // upload this file
    let resp = client.files.create(file_req.clone()).await?;
    // delete from group 4 and check that it was removed
    let group4 = groups.pop().unwrap();
    client
        .files
        .delete(
            &resp.sha256,
            &resp.id,
            &FileDeleteOpts::default().groups(vec![group4.clone()]),
        )
        .await?;
    let sample = client.files.get(&resp.sha256).await?;
    is_not_in!(&sample.groups(), group4.as_str());
    // delete from groups 2 and 3 and check that they were removed;
    // add group4, which was already deleted, but the request should go through anyway
    let group3 = groups.pop().unwrap();
    let group2 = groups.pop().unwrap();
    let del_groups = vec![group2.clone(), group3.clone(), group4];
    let del_opts = FileDeleteOpts::default().groups(del_groups);
    client
        .files
        .delete(&resp.sha256, &resp.id, &del_opts)
        .await?;
    let sample = client.files.get(&resp.sha256).await?;
    is_not_in!(&sample.groups(), group2.as_str());
    is_not_in!(&sample.groups(), group3.as_str());
    // attempt to delete from groups that all do not have access
    let result = client.files.delete(&resp.sha256, &resp.id, &del_opts).await;
    fail!(result, 404);
    // attempt to delete from nonexistent groups
    let fake_groups = vec!["fake".to_owned(), "groups".to_owned()];
    let fake_del_opts = FileDeleteOpts::default().groups(fake_groups);
    let result = client
        .files
        .delete(&resp.sha256, &resp.id, &fake_del_opts)
        .await;
    // check for error
    fail!(result, 404);
    // create a new user and attempt to delete without being an owner/manager in the group
    let user_client = generators::client(&client).await?;
    let user_username = user_client.users.info().await?.username;
    // add user to group as a user (not a manager)
    let group_update =
        GroupUpdate::default().users(GroupUsersUpdate::default().direct_add(&user_username));
    client
        .groups
        .update(groups.first().unwrap(), &group_update)
        .await?;
    let result = user_client
        .files
        .delete(
            &resp.sha256,
            &resp.id,
            &FileDeleteOpts::default().groups(groups.clone()),
        )
        .await;
    fail!(result, 401);
    // finally set user as a manager in the group and delete the submission as the user
    let group_update =
        GroupUpdate::default().managers(GroupUsersUpdate::default().direct_add(&user_username));
    client
        .groups
        .update(groups.first().unwrap(), &group_update)
        .await?;
    user_client
        .files
        .delete(
            &resp.sha256,
            &resp.id,
            &FileDeleteOpts::default().groups(groups),
        )
        .await?;
    Ok(())
}

/// Test if submitter tag is deleted after submission deletion
#[tokio::test]
async fn delete_submitter_tag() -> Result<(), thorium::Error> {
    // get admin client
    let admin_client = test_utilities::admin_client().await?;
    // Create a group
    let admin_group = generators::groups(1, &admin_client).await?.remove(0).name;
    // build a sample request
    let file_req = SampleRequest::new_buffer(Buffer::new("highwind"), vec![admin_group.clone()]);
    // upload this file
    let sha256 = admin_client.files.create(file_req.clone()).await?.sha256;

    // create a new user and generate a client for that user
    let user_client = generators::client(&admin_client).await?;
    // retrieve both users
    let admin_username = admin_client.users.info().await?.username;
    let user_username = user_client.users.info().await?.username;
    // create a group for that user
    let user_group = generators::groups(1, &user_client).await?.remove(0).name;
    // add user to admin's group
    let group_update =
        GroupUpdate::default().users(GroupUsersUpdate::default().direct_add(&user_username));
    admin_client
        .groups
        .update(&admin_group, &group_update)
        .await?;
    // upload an identical file as a new submission from the new user in both groups
    let file_req = SampleRequest::new_buffer(
        Buffer::new("highwind"),
        vec![admin_group.clone(), user_group.clone()],
    );
    let id1 = user_client.files.create(file_req).await?.id;
    // upload again but only in user group
    let file_req = SampleRequest::new_buffer(Buffer::new("highwind"), vec![user_group.clone()]);
    let id2 = user_client.files.create(file_req).await?.id;

    // check that both users are tagged as submitters in their respective groups
    let submitter_key = "submitter";
    let sample = admin_client.files.get(&sha256).await?;
    has_tag!(&sample.tags, submitter_key, &admin_username, &admin_group);
    has_tag!(&sample.tags, submitter_key, &user_username, &user_group);
    has_tag!(&sample.tags, submitter_key, &user_username, &admin_group);
    // delete the user's second submission
    user_client
        .files
        .delete(&sha256, &id2, &FileDeleteOpts::default())
        .await?;
    // wait for tags to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    // check that the second user's tags still exist, as they still have a submission in both groups
    let sample = admin_client.files.get(&sha256).await?;
    has_tag!(&sample.tags, submitter_key, &user_username, &user_group);
    has_tag!(&sample.tags, submitter_key, &user_username, &admin_group);
    // delete the user's first submission from user group
    user_client
        .files
        .delete(
            &sha256,
            &id1,
            &FileDeleteOpts::default().groups(vec![user_group.clone()]),
        )
        .await?;
    // wait for tags to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    // check that the user's tag for the user group is deleted
    let sample = admin_client.files.get(&sha256).await?;
    no_tag!(&sample.tags, submitter_key, &user_username, &user_group);
    // check that the other submitter tags still exists
    has_tag!(&sample.tags, submitter_key, &admin_username, &admin_group);
    has_tag!(&sample.tags, submitter_key, &user_username, &admin_group);
    // delete the user's last submission and check that user is deleted as a submitter
    user_client
        .files
        .delete(
            &sha256,
            &id1,
            &FileDeleteOpts::default().groups(vec![admin_group.clone()]),
        )
        .await?;
    // wait for tags to update
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let sample = admin_client.files.get(&sha256).await?;
    no_tag!(&sample.tags, submitter_key, &user_username);
    has_tag!(&sample.tags, submitter_key, &admin_username);

    Ok(())
}
