//! Test association routes

use thorium::models::{
    Association, AssociationDeleteRequest, AssociationKind, AssociationListOpts,
    AssociationRequest, AssociationTarget, Cursor, Directionality, EntityMetadataRequest,
    EntityRequest, GroupUpdate, GroupUsersUpdate, RepoCheckout, RepoRequest,
};
use thorium::test_utilities::{self, generators};
use thorium::{Thorium, fail, is};
use uuid::Uuid;

/// The roles a test user can have in a group
enum Role {
    /// A regular user in a group
    User,
    /// A manager of a group
    Manager,
}

/// Upload a random sample and get its sha256
///
/// # Arguments
///
/// * `client` - The client to upload this sample with
/// * `groups` - The groups to upload this sample to
async fn upload(client: &Thorium, groups: &[&str]) -> Result<String, thorium::Error> {
    // build a random sample in our first group
    let mut req = generators::gen_sample(groups[0]);
    // add any extra groups
    req.groups = groups.iter().map(|group| (*group).to_owned()).collect();
    // upload this sample
    Ok(client.files.create(req).await?.sha256)
}

/// Create a random repo and get its normalized url
///
/// # Arguments
///
/// * `client` - The client to create this repo with
/// * `group` - The group to create this repo in
async fn repo(client: &Thorium, group: &str) -> Result<String, thorium::Error> {
    // build a random repo in this group
    let req = generators::gen_repo(group);
    // create this repo
    Ok(client.repos.create(&req).await?.url)
}

/// Create an `Other` entity and get it as an association target
///
/// `Other` entities are used so no implicit vendor/device associations are created.
///
/// # Arguments
///
/// * `client` - The client to create this entity with
/// * `group` - The group to create this entity in
async fn entity(client: &Thorium, group: &str) -> Result<AssociationTarget, thorium::Error> {
    // build a random name for this entity
    let name = generators::gen_string(24);
    // build the request for this entity
    let req = EntityRequest::new(&name, EntityMetadataRequest::Other, vec![group]);
    // create this entity
    let resp = client.entities.create(req).await?;
    Ok(AssociationTarget::Entity { id: resp.id, name })
}

/// Create a new user with a specific role in some groups
///
/// # Arguments
///
/// * `admin` - An admin client
/// * `groups` - The groups and the role to give this user in each of them
async fn member(
    admin: &Thorium,
    groups: &[(&str, Role)],
) -> Result<(Thorium, String), thorium::Error> {
    // create a new user
    let client = generators::client(admin).await?;
    // get this users name
    let username = client.users.info().await?.username;
    // add this user to each group
    for (group, role) in groups {
        // build the users to add to this group
        let users = GroupUsersUpdate::default().direct_add(&username);
        // build the update for this group
        let update = match role {
            Role::User => GroupUpdate::default().users(users),
            Role::Manager => GroupUpdate::default().managers(users),
        };
        // add this user to this group
        admin.groups.update(group, &update).await?;
    }
    Ok((client, username))
}

/// Collect all of the associations in a cursor
///
/// # Arguments
///
/// * `cursor` - The cursor to exhaust
async fn collect(mut cursor: Cursor<Association>) -> Result<Vec<Association>, thorium::Error> {
    // keep a list of all of our associations
    let mut assocs = Vec::default();
    // page through this cursor
    loop {
        // add this page of associations
        assocs.append(&mut cursor.data);
        // check if this cursor is exhausted
        if cursor.exhausted() {
            break;
        }
        // get the next page of associations
        cursor.refill().await?;
    }
    Ok(assocs)
}

/// List all associations for some target
///
/// # Arguments
///
/// * `client` - The client to list associations with
/// * `target` - The target to list associations for
async fn list(
    client: &Thorium,
    target: &AssociationTarget,
) -> Result<Vec<Association>, thorium::Error> {
    // use the default list options
    let opts = AssociationListOpts::default();
    // list associations for the right kind of target
    let cursor = match target {
        AssociationTarget::File(sha256) => client.associations.list_file(sha256, &opts).await?,
        AssociationTarget::Repo(url) => client.associations.list_repo(url, &opts).await?,
        AssociationTarget::Entity { id, .. } => client.associations.list_entity(*id, &opts).await?,
    };
    collect(cursor).await
}

/// Find the associations of a specific kind with a specific target
///
/// # Arguments
///
/// * `assocs` - The associations to search
/// * `kind` - The kind of association to find
/// * `other` - The other side of the association to find
fn find<'a>(
    assocs: &'a [Association],
    kind: AssociationKind,
    other: &AssociationTarget,
) -> Vec<&'a Association> {
    assocs
        .iter()
        .filter(|assoc| assoc.kind == kind && &assoc.other == other)
        .collect()
}

/// Make sure exactly one association of a kind exists with a target and get it
///
/// # Arguments
///
/// * `client` - The client to list associations with
/// * `source` - The target to list associations for
/// * `kind` - The kind of association to find
/// * `other` - The other side of the association to find
async fn get_one(
    client: &Thorium,
    source: &AssociationTarget,
    kind: AssociationKind,
    other: &AssociationTarget,
) -> Result<Association, thorium::Error> {
    // list the associations for our source
    let assocs = list(client, source).await?;
    // find the associations we are looking for
    let found = find(&assocs, kind, other);
    // make sure we found exactly one
    is!(
        found.len(),
        1,
        format!("{kind} associations with {other:?}")
    );
    Ok(found[0].clone())
}

/// Make sure no associations of a kind exist with a target
///
/// # Arguments
///
/// * `client` - The client to list associations with
/// * `source` - The target to list associations for
/// * `kind` - The kind of association to look for
/// * `other` - The other side of the association to look for
async fn none(
    client: &Thorium,
    source: &AssociationTarget,
    kind: AssociationKind,
    other: &AssociationTarget,
) -> Result<(), thorium::Error> {
    // list the associations for our source
    let assocs = list(client, source).await?;
    // make sure we didn't find any matching associations
    is!(
        find(&assocs, kind, other).len(),
        0,
        format!("{kind} associations with {other:?}")
    );
    Ok(())
}

/// Create a single association between two targets
///
/// # Arguments
///
/// * `client` - The client to create this association with
/// * `kind` - The kind of association to create
/// * `source` - The source of this association
/// * `target` - The target of this association
async fn associate(
    client: &Thorium,
    kind: AssociationKind,
    source: &AssociationTarget,
    target: &AssociationTarget,
) -> Result<(), thorium::Error> {
    // build our association request
    let req = AssociationRequest::new(kind, source.clone()).target(target.clone());
    // create this association
    client.associations.create(&req).await?;
    Ok(())
}

/// Build a random sha256 that doesn't exist in Thorium
fn fake_sha256() -> String {
    // two uuids without dashes is 64 hex characters
    format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
}

/// Test creating an association between two files
#[tokio::test]
async fn create_file_to_file() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload two samples
    let a = AssociationTarget::File(upload(&client, &[&group]).await?);
    let b = AssociationTarget::File(upload(&client, &[&group]).await?);
    // associate our samples
    associate(&client, AssociationKind::AssociatedWith, &a, &b).await?;
    // make sure our source side has the association
    let assoc = get_one(&client, &a, AssociationKind::AssociatedWith, &b).await?;
    is!(assoc.direction, Directionality::To);
    is!(assoc.submitter, "thorium");
    is!(assoc.groups, vec![group.clone()]);
    // make sure our target side has the inverted association
    let assoc = get_one(&client, &b, AssociationKind::AssociatedWith, &a).await?;
    is!(assoc.direction, Directionality::From);
    Ok(())
}

/// Test creating a bidirectional association
#[tokio::test]
async fn create_bidirectional() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload two samples
    let a = AssociationTarget::File(upload(&client, &[&group]).await?);
    let b = AssociationTarget::File(upload(&client, &[&group]).await?);
    // create a bidirectional association
    let req = AssociationRequest::new(AssociationKind::AssociatedWith, a.clone())
        .target(b.clone())
        .biderectional();
    client.associations.create(&req).await?;
    // make sure both sides are bidirectional
    let assoc = get_one(&client, &a, AssociationKind::AssociatedWith, &b).await?;
    is!(assoc.direction, Directionality::Bidirectional);
    let assoc = get_one(&client, &b, AssociationKind::AssociatedWith, &a).await?;
    is!(assoc.direction, Directionality::Bidirectional);
    Ok(())
}

/// Test creating associations with file, repo, and entity targets at once
#[tokio::test]
async fn create_mixed_targets() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create our source and targets
    let source = AssociationTarget::File(upload(&client, &[&group]).await?);
    let file = AssociationTarget::File(upload(&client, &[&group]).await?);
    let repo = AssociationTarget::Repo(repo(&client, &group).await?);
    let entity = entity(&client, &group).await?;
    // associate our source with all of our targets
    let req = AssociationRequest::new(AssociationKind::FileFor, source.clone())
        .target(file.clone())
        .target(repo.clone())
        .target(entity.clone());
    client.associations.create(&req).await?;
    // make sure each association exists on our source
    let assocs = list(&client, &source).await?;
    for target in [&file, &repo, &entity] {
        is!(
            find(&assocs, AssociationKind::FileFor, target).len(),
            1,
            format!("association with {target:?}")
        );
    }
    // make sure each target sees our source
    for target in [&file, &repo, &entity] {
        let assoc = get_one(&client, target, AssociationKind::FileFor, &source).await?;
        is!(assoc.direction, Directionality::From);
    }
    Ok(())
}

/// Test creating an association with explicit groups
#[tokio::test]
async fn create_explicit_groups() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create two groups
    let mut groups = generators::groups(2, &client).await?;
    let g1 = groups.remove(0).name;
    let g2 = groups.remove(0).name;
    // upload two samples in both groups
    let a = AssociationTarget::File(upload(&client, &[&g1, &g2]).await?);
    let b = AssociationTarget::File(upload(&client, &[&g1, &g2]).await?);
    // associate our samples in only the first group
    let req = AssociationRequest::new(AssociationKind::AssociatedWith, a.clone())
        .target(b.clone())
        .groups(vec![g1.clone()]);
    client.associations.create(&req).await?;
    // make sure our association is only in our first group
    let assoc = get_one(&client, &a, AssociationKind::AssociatedWith, &b).await?;
    is!(assoc.groups, vec![g1.clone()]);
    Ok(())
}

/// Test that creating an association in a group we are not in fails
#[tokio::test]
async fn create_group_not_member() -> Result<(), thorium::Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // create two groups
    let mut groups = generators::groups(2, &admin).await?;
    let g1 = groups.remove(0).name;
    let g2 = groups.remove(0).name;
    // create a user that is only in our first group
    let (user, _) = member(&admin, &[(&g1, Role::User)]).await?;
    // upload two samples
    let a = AssociationTarget::File(upload(&user, &[&g1]).await?);
    let b = AssociationTarget::File(upload(&user, &[&g1]).await?);
    // try to associate our samples in a group we are not in
    let req = AssociationRequest::new(AssociationKind::AssociatedWith, a)
        .target(b)
        .groups(vec![g2]);
    let resp = user.associations.create(&req).await;
    fail!(resp, 401);
    Ok(())
}

/// Test that creating an association from a missing source fails
#[tokio::test]
async fn create_missing_source() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload a target sample
    let b = AssociationTarget::File(upload(&client, &[&group]).await?);
    // try to associate a sample that doesn't exist
    let a = AssociationTarget::File(fake_sha256());
    let req = AssociationRequest::new(AssociationKind::AssociatedWith, a).target(b);
    let resp = client.associations.create(&req).await;
    fail!(resp, 404);
    Ok(())
}

/// Test that the submitter of an association can delete it
#[tokio::test]
async fn delete_by_submitter() -> Result<(), thorium::Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &admin).await?.remove(0).name;
    // create a regular user in this group
    let (user, _) = member(&admin, &[(&group, Role::User)]).await?;
    // upload two samples and associate them
    let a = AssociationTarget::File(upload(&user, &[&group]).await?);
    let b = AssociationTarget::File(upload(&user, &[&group]).await?);
    associate(&user, AssociationKind::AssociatedWith, &a, &b).await?;
    // delete this association
    let req =
        AssociationDeleteRequest::new(AssociationKind::AssociatedWith, a.clone()).target(b.clone());
    user.associations.delete(&req).await?;
    // make sure both sides of this association are gone
    none(&user, &a, AssociationKind::AssociatedWith, &b).await?;
    none(&user, &b, AssociationKind::AssociatedWith, &a).await?;
    Ok(())
}

/// Test that a regular user cannot delete another users association
#[tokio::test]
async fn delete_non_submitter_rejected() -> Result<(), thorium::Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &admin).await?.remove(0).name;
    // create two regular users in this group
    let (user1, _) = member(&admin, &[(&group, Role::User)]).await?;
    let (user2, _) = member(&admin, &[(&group, Role::User)]).await?;
    // upload two samples and associate them as our first user
    let a = AssociationTarget::File(upload(&user1, &[&group]).await?);
    let b = AssociationTarget::File(upload(&user1, &[&group]).await?);
    associate(&user1, AssociationKind::AssociatedWith, &a, &b).await?;
    // try to delete this association as our second user
    let req =
        AssociationDeleteRequest::new(AssociationKind::AssociatedWith, a.clone()).target(b.clone());
    let resp = user2.associations.delete(&req).await;
    fail!(
        resp,
        401,
        "Cannot delete associations submitted by other users"
    );
    // make sure our association still exists
    get_one(&user1, &a, AssociationKind::AssociatedWith, &b).await?;
    Ok(())
}

/// Test that a group manager can delete another users association
#[tokio::test]
async fn delete_by_group_manager() -> Result<(), thorium::Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &admin).await?.remove(0).name;
    // create a regular user and a manager in this group
    let (user, _) = member(&admin, &[(&group, Role::User)]).await?;
    let (manager, _) = member(&admin, &[(&group, Role::Manager)]).await?;
    // upload two samples and associate them as our user
    let a = AssociationTarget::File(upload(&user, &[&group]).await?);
    let b = AssociationTarget::File(upload(&user, &[&group]).await?);
    associate(&user, AssociationKind::AssociatedWith, &a, &b).await?;
    // delete this association as our manager
    let req =
        AssociationDeleteRequest::new(AssociationKind::AssociatedWith, a.clone()).target(b.clone());
    manager.associations.delete(&req).await?;
    // make sure both sides of this association are gone
    none(&user, &a, AssociationKind::AssociatedWith, &b).await?;
    none(&user, &b, AssociationKind::AssociatedWith, &a).await?;
    Ok(())
}

/// Test that an admin can delete another users association
#[tokio::test]
async fn delete_by_admin() -> Result<(), thorium::Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &admin).await?.remove(0).name;
    // create a regular user in this group
    let (user, _) = member(&admin, &[(&group, Role::User)]).await?;
    // upload two samples and associate them as our user
    let a = AssociationTarget::File(upload(&user, &[&group]).await?);
    let b = AssociationTarget::File(upload(&user, &[&group]).await?);
    associate(&user, AssociationKind::AssociatedWith, &a, &b).await?;
    // delete this association as an admin
    let req =
        AssociationDeleteRequest::new(AssociationKind::AssociatedWith, a.clone()).target(b.clone());
    admin.associations.delete(&req).await?;
    // make sure both sides of this association are gone
    none(&user, &a, AssociationKind::AssociatedWith, &b).await?;
    none(&user, &b, AssociationKind::AssociatedWith, &a).await?;
    Ok(())
}

/// Test deleting an association when we can only delete it from some of its groups
#[tokio::test]
async fn delete_partial_permission() -> Result<(), thorium::Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // create two groups
    let mut groups = generators::groups(2, &admin).await?;
    let g1 = groups.remove(0).name;
    let g2 = groups.remove(0).name;
    // create a regular user in both groups
    let (user1, _) = member(&admin, &[(&g1, Role::User), (&g2, Role::User)]).await?;
    // create a user that manages our first group but is a regular user in our second
    let (user2, _) = member(&admin, &[(&g1, Role::Manager), (&g2, Role::User)]).await?;
    // upload two samples in both groups and associate them as our first user
    let a = AssociationTarget::File(upload(&user1, &[&g1, &g2]).await?);
    let b = AssociationTarget::File(upload(&user1, &[&g1, &g2]).await?);
    associate(&user1, AssociationKind::AssociatedWith, &a, &b).await?;
    // try to delete this association from all groups as our second user
    let req =
        AssociationDeleteRequest::new(AssociationKind::AssociatedWith, a.clone()).target(b.clone());
    let resp = user2.associations.delete(&req).await;
    fail!(resp, 401);
    // make sure nothing was deleted
    let assoc = get_one(&user1, &a, AssociationKind::AssociatedWith, &b).await?;
    is!(assoc.groups.len(), 2);
    // delete this association from just the group we manage
    let req = req.groups(vec![g1.clone()]);
    user2.associations.delete(&req).await?;
    // make sure this association is only left in our second group
    let assoc = get_one(&user1, &a, AssociationKind::AssociatedWith, &b).await?;
    is!(assoc.groups, vec![g2.clone()]);
    let assoc = get_one(&user1, &b, AssociationKind::AssociatedWith, &a).await?;
    is!(assoc.groups, vec![g2.clone()]);
    Ok(())
}

/// Test that deleting associations that don't exist fails
#[tokio::test]
async fn delete_no_match() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload three samples and associate two of them
    let a = AssociationTarget::File(upload(&client, &[&group]).await?);
    let b = AssociationTarget::File(upload(&client, &[&group]).await?);
    let c = AssociationTarget::File(upload(&client, &[&group]).await?);
    associate(&client, AssociationKind::AssociatedWith, &a, &b).await?;
    // try to delete an association with the wrong kind
    let req = AssociationDeleteRequest::new(AssociationKind::FileFor, a.clone()).target(b.clone());
    let resp = client.associations.delete(&req).await;
    fail!(resp, 404, "No matching associations found");
    // try to delete an association with a target that was never associated
    let req = AssociationDeleteRequest::new(AssociationKind::AssociatedWith, a.clone()).target(c);
    let resp = client.associations.delete(&req).await;
    fail!(resp, 404, "No matching associations found");
    // make sure our original association still exists
    get_one(&client, &a, AssociationKind::AssociatedWith, &b).await?;
    Ok(())
}

/// Test that deleting associations without any targets fails
#[tokio::test]
async fn delete_empty_targets() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload a sample
    let a = AssociationTarget::File(upload(&client, &[&group]).await?);
    // try to delete associations without any targets
    let req = AssociationDeleteRequest::new(AssociationKind::AssociatedWith, a);
    let resp = client.associations.delete(&req).await;
    fail!(resp, 400);
    Ok(())
}

/// Test that deleting associations from a missing source fails
#[tokio::test]
async fn delete_missing_source() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload a target sample
    let b = AssociationTarget::File(upload(&client, &[&group]).await?);
    // try to delete associations from a sample that doesn't exist
    let a = AssociationTarget::File(fake_sha256());
    let req = AssociationDeleteRequest::new(AssociationKind::AssociatedWith, a).target(b);
    let resp = client.associations.delete(&req).await;
    fail!(resp, 404);
    Ok(())
}

/// Test that deleting a bidirectional association removes both sides
#[tokio::test]
async fn delete_bidirectional() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload two samples
    let a = AssociationTarget::File(upload(&client, &[&group]).await?);
    let b = AssociationTarget::File(upload(&client, &[&group]).await?);
    // create a bidirectional association
    let req = AssociationRequest::new(AssociationKind::AssociatedWith, a.clone())
        .target(b.clone())
        .biderectional();
    client.associations.create(&req).await?;
    // delete this association
    let req =
        AssociationDeleteRequest::new(AssociationKind::AssociatedWith, a.clone()).target(b.clone());
    client.associations.delete(&req).await?;
    // make sure both sides of this association are gone
    none(&client, &a, AssociationKind::AssociatedWith, &b).await?;
    none(&client, &b, AssociationKind::AssociatedWith, &a).await?;
    Ok(())
}

/// Test deleting an association starting from its target
#[tokio::test]
async fn delete_from_target_side() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload two samples and associate them
    let a = AssociationTarget::File(upload(&client, &[&group]).await?);
    let b = AssociationTarget::File(upload(&client, &[&group]).await?);
    associate(&client, AssociationKind::AssociatedWith, &a, &b).await?;
    // delete this association using our target as the source
    let req =
        AssociationDeleteRequest::new(AssociationKind::AssociatedWith, b.clone()).target(a.clone());
    client.associations.delete(&req).await?;
    // make sure both sides of this association are gone
    none(&client, &a, AssociationKind::AssociatedWith, &b).await?;
    none(&client, &b, AssociationKind::AssociatedWith, &a).await?;
    Ok(())
}

/// Test restricting a delete to a specific direction
#[tokio::test]
async fn delete_direction_filter() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload two samples and associate them
    let a = AssociationTarget::File(upload(&client, &[&group]).await?);
    let b = AssociationTarget::File(upload(&client, &[&group]).await?);
    associate(&client, AssociationKind::AssociatedWith, &a, &b).await?;
    // try to delete only associations pointing at our source
    let req = AssociationDeleteRequest::new(AssociationKind::AssociatedWith, a.clone())
        .target(b.clone())
        .direction(Directionality::From);
    let resp = client.associations.delete(&req).await;
    fail!(resp, 404);
    // make sure our association still exists
    get_one(&client, &a, AssociationKind::AssociatedWith, &b).await?;
    // delete only associations pointing away from our source
    let req = AssociationDeleteRequest::new(AssociationKind::AssociatedWith, a.clone())
        .target(b.clone())
        .direction(Directionality::To);
    client.associations.delete(&req).await?;
    // make sure both sides of this association are gone
    none(&client, &a, AssociationKind::AssociatedWith, &b).await?;
    none(&client, &b, AssociationKind::AssociatedWith, &a).await?;
    Ok(())
}

/// Test that deleting an association leaves other associations alone
#[tokio::test]
async fn delete_leaves_other_targets() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload three samples
    let a = AssociationTarget::File(upload(&client, &[&group]).await?);
    let b = AssociationTarget::File(upload(&client, &[&group]).await?);
    let c = AssociationTarget::File(upload(&client, &[&group]).await?);
    // associate our first sample with the other two
    let req = AssociationRequest::new(AssociationKind::AssociatedWith, a.clone())
        .target(b.clone())
        .target(c.clone());
    client.associations.create(&req).await?;
    // also create a different kind of association with our second sample
    associate(&client, AssociationKind::FileFor, &a, &b).await?;
    // delete only the association with our second sample
    let req =
        AssociationDeleteRequest::new(AssociationKind::AssociatedWith, a.clone()).target(b.clone());
    client.associations.delete(&req).await?;
    // make sure only that association was removed
    none(&client, &a, AssociationKind::AssociatedWith, &b).await?;
    get_one(&client, &a, AssociationKind::AssociatedWith, &c).await?;
    get_one(&client, &a, AssociationKind::FileFor, &b).await?;
    Ok(())
}

/// Test deleting associations with repo and entity targets
#[tokio::test]
async fn delete_repo_and_entity_targets() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create our source and targets
    let source = AssociationTarget::File(upload(&client, &[&group]).await?);
    let repo = AssociationTarget::Repo(repo(&client, &group).await?);
    let entity = entity(&client, &group).await?;
    // associate our source with our targets
    let req = AssociationRequest::new(AssociationKind::FileFor, source.clone())
        .target(repo.clone())
        .target(entity.clone());
    client.associations.create(&req).await?;
    // entity names are ignored when deleting so use an empty one
    let entity_no_name = match &entity {
        AssociationTarget::Entity { id, .. } => AssociationTarget::Entity {
            id: *id,
            name: String::new(),
        },
        _ => unreachable!("entity helper always returns an entity target"),
    };
    // delete both associations
    let req = AssociationDeleteRequest::new(AssociationKind::FileFor, source.clone())
        .targets(vec![repo.clone(), entity_no_name]);
    client.associations.delete(&req).await?;
    // make sure all sides of these associations are gone
    none(&client, &source, AssociationKind::FileFor, &repo).await?;
    none(&client, &source, AssociationKind::FileFor, &entity).await?;
    none(&client, &repo, AssociationKind::FileFor, &source).await?;
    none(&client, &entity, AssociationKind::FileFor, &source).await?;
    Ok(())
}

/// Test that listing associations for a missing file fails
#[tokio::test]
async fn list_file_missing() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // try to list associations for a file that doesn't exist
    let resp = client
        .associations
        .list_file(&fake_sha256(), &AssociationListOpts::default())
        .await
        .map(|_| ());
    fail!(resp, 404);
    Ok(())
}

/// Test that listing associations for a file we can't see fails
#[tokio::test]
async fn list_file_inaccessible() -> Result<(), thorium::Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // create two groups
    let mut groups = generators::groups(2, &admin).await?;
    let g1 = groups.remove(0).name;
    let g2 = groups.remove(0).name;
    // create a user that is only in our second group
    let (user, _) = member(&admin, &[(&g2, Role::User)]).await?;
    // upload a sample to our first group
    let sha256 = upload(&admin, &[&g1]).await?;
    // try to list associations for this file as our user
    let resp = user
        .associations
        .list_file(&sha256, &AssociationListOpts::default())
        .await
        .map(|_| ());
    fail!(resp, 404);
    Ok(())
}

/// Test listing a files associations from specific groups
#[tokio::test]
async fn list_file_groups_filter() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create two groups
    let mut groups = generators::groups(2, &client).await?;
    let g1 = groups.remove(0).name;
    let g2 = groups.remove(0).name;
    // upload three samples in both groups
    let a = AssociationTarget::File(upload(&client, &[&g1, &g2]).await?);
    let b = AssociationTarget::File(upload(&client, &[&g1, &g2]).await?);
    let c = AssociationTarget::File(upload(&client, &[&g1, &g2]).await?);
    // associate our samples in different groups
    let req = AssociationRequest::new(AssociationKind::AssociatedWith, a.clone())
        .target(b.clone())
        .groups(vec![g1.clone()]);
    client.associations.create(&req).await?;
    let req = AssociationRequest::new(AssociationKind::AssociatedWith, a.clone())
        .target(c.clone())
        .groups(vec![g2.clone()]);
    client.associations.create(&req).await?;
    // list our associations from only our first group
    let AssociationTarget::File(sha256) = &a else {
        unreachable!("a is always a file")
    };
    let opts = AssociationListOpts::default().groups(vec![g1]);
    let cursor = client.associations.list_file(sha256, &opts).await?;
    let assocs = collect(cursor).await?;
    // make sure we only got the association in our first group
    is!(find(&assocs, AssociationKind::AssociatedWith, &b).len(), 1);
    is!(find(&assocs, AssociationKind::AssociatedWith, &c).len(), 0);
    Ok(())
}

/// Test paging through a files associations
#[tokio::test]
async fn list_file_paging() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // upload a source sample
    let sha256 = upload(&client, &[&group]).await?;
    let source = AssociationTarget::File(sha256.clone());
    // associate our source with 6 targets
    let mut req = AssociationRequest::new(AssociationKind::AssociatedWith, source);
    for _ in 0..6 {
        let target = AssociationTarget::File(upload(&client, &[&group]).await?);
        req = req.target(target);
    }
    client.associations.create(&req).await?;
    // page through all of our associations 2 at a time
    let opts = AssociationListOpts::default().page_size(2);
    let cursor = client.associations.list_file(&sha256, &opts).await?;
    let assocs = collect(cursor).await?;
    is!(assocs.len(), 6);
    // make sure our limit is respected
    let opts = AssociationListOpts::default().page_size(2).limit(3);
    let cursor = client.associations.list_file(&sha256, &opts).await?;
    let assocs = collect(cursor).await?;
    is!(assocs.len(), 3);
    Ok(())
}

/// Test listing the associations for a repo
#[tokio::test]
async fn list_repo() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a repo, a file, and an entity
    let repo = AssociationTarget::Repo(repo(&client, &group).await?);
    let file = AssociationTarget::File(upload(&client, &[&group]).await?);
    let entity = entity(&client, &group).await?;
    // associate our file with our repo and our repo with our entity
    associate(&client, AssociationKind::FileFor, &file, &repo).await?;
    associate(&client, AssociationKind::DevelopedBy, &repo, &entity).await?;
    // make sure our repo sees both associations in the right direction
    let assocs = list(&client, &repo).await?;
    let found = find(&assocs, AssociationKind::FileFor, &file);
    is!(found.len(), 1);
    is!(found[0].direction, Directionality::From);
    let found = find(&assocs, AssociationKind::DevelopedBy, &entity);
    is!(found.len(), 1);
    is!(found[0].direction, Directionality::To);
    Ok(())
}

/// Test listing the associations for a nested repo
#[tokio::test]
async fn list_repo_nested_path() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create a nested repo
    let url = format!(
        "provider.tld/{}/{}/{}",
        generators::gen_string(12),
        generators::gen_string(12),
        generators::gen_string(12)
    );
    let req = RepoRequest::new(&url, vec![&group], Some(RepoCheckout::branch("main")));
    let repo = AssociationTarget::Repo(client.repos.create(&req).await?.url);
    // upload a file and associate it with our nested repo
    let file = AssociationTarget::File(upload(&client, &[&group]).await?);
    associate(&client, AssociationKind::FileFor, &file, &repo).await?;
    // make sure we can list our nested repos associations
    let assoc = get_one(&client, &repo, AssociationKind::FileFor, &file).await?;
    is!(assoc.direction, Directionality::From);
    Ok(())
}

/// Test that listing associations for a missing repo fails
#[tokio::test]
async fn list_repo_missing() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // build a repo url that doesn't exist
    let url = format!(
        "provider.tld/{}/{}",
        generators::gen_string(24),
        generators::gen_string(24)
    );
    // try to list associations for this repo
    let resp = client
        .associations
        .list_repo(&url, &AssociationListOpts::default())
        .await
        .map(|_| ());
    fail!(resp, 404);
    Ok(())
}

/// Test that listing associations for a repo we can't see fails
#[tokio::test]
async fn list_repo_inaccessible() -> Result<(), thorium::Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // create two groups
    let mut groups = generators::groups(2, &admin).await?;
    let g1 = groups.remove(0).name;
    let g2 = groups.remove(0).name;
    // create a user that is only in our second group
    let (user, _) = member(&admin, &[(&g2, Role::User)]).await?;
    // create a repo in our first group
    let url = repo(&admin, &g1).await?;
    // try to list associations for this repo as our user
    let resp = user
        .associations
        .list_repo(&url, &AssociationListOpts::default())
        .await
        .map(|_| ());
    fail!(resp, 404);
    Ok(())
}

/// Test listing the associations for an entity
#[tokio::test]
async fn list_entity() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // create an entity and a file
    let entity = entity(&client, &group).await?;
    let file = AssociationTarget::File(upload(&client, &[&group]).await?);
    // associate our entity with our file
    associate(&client, AssociationKind::FlagFor, &entity, &file).await?;
    // make sure our entity sees this association
    let assoc = get_one(&client, &entity, AssociationKind::FlagFor, &file).await?;
    is!(assoc.direction, Directionality::To);
    // make sure our file sees our entity with its name
    let assoc = get_one(&client, &file, AssociationKind::FlagFor, &entity).await?;
    is!(assoc.direction, Directionality::From);
    Ok(())
}

/// Test that listing associations for a missing entity fails
#[tokio::test]
async fn list_entity_missing() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // try to list associations for an entity that doesn't exist
    let resp = client
        .associations
        .list_entity(Uuid::new_v4(), &AssociationListOpts::default())
        .await
        .map(|_| ());
    fail!(resp, 404);
    Ok(())
}

/// Test that listing associations for an entity we can't see fails
#[tokio::test]
async fn list_entity_inaccessible() -> Result<(), thorium::Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // create two groups
    let mut groups = generators::groups(2, &admin).await?;
    let g1 = groups.remove(0).name;
    let g2 = groups.remove(0).name;
    // create a user that is only in our second group
    let (user, _) = member(&admin, &[(&g2, Role::User)]).await?;
    // create an entity in our first group
    let AssociationTarget::Entity { id, .. } = entity(&admin, &g1).await? else {
        unreachable!("entity helper always returns an entity target")
    };
    // try to list associations for this entity as our user
    let resp = user
        .associations
        .list_entity(id, &AssociationListOpts::default())
        .await
        .map(|_| ());
    fail!(resp, 404);
    Ok(())
}
