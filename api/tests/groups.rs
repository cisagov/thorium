//! Tests the Groups routes in Thorium

use http::StatusCode;
use thorium::models::{GroupUpdate, GroupUsersRequest, GroupUsersUpdate, NetworkPolicyListOpts};
use thorium::test_utilities::{self, generators};
use thorium::{fail, is, is_in, is_not_in, vec_in_vec};

#[tokio::test]
async fn create() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create the users to add to this group
    let managers = generators::users(2, &client).await?;
    let users = generators::users(2, &client).await?;
    let monitors = generators::users(2, &client).await?;
    // build a random group request
    let group_req = generators::gen_group()
        // add the users we created
        .managers(
            GroupUsersRequest::default()
                .direct(&managers[0])
                .direct(&managers[1]),
        )
        .users(
            GroupUsersRequest::default()
                .direct(&users[0])
                .direct(&users[1]),
        )
        .monitors(
            GroupUsersRequest::default()
                .direct(&monitors[0])
                .direct(&monitors[1]),
        );
    let resp = client.groups.create(&group_req).await?;
    is!(resp.status().as_u16(), 204);
    // make sure our group request is in our created groups
    let resp = client.groups.list().page(100).details().exec().await?;
    is_in!(resp.details, group_req);
    // retrieve group and check that the request matches
    let retrieved = client.groups.get(&group_req.name).await?;
    is!(retrieved, group_req);
    Ok(())
}

#[tokio::test]
async fn create_conflict() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create the users to add to this group
    let managers = generators::users(2, &client).await?;
    let users = generators::users(2, &client).await?;
    let monitors = generators::users(2, &client).await?;
    // build a random group request
    let mut group_req = generators::gen_group()
        // add the users we created
        .managers(
            GroupUsersRequest::default()
                .direct(&managers[0])
                .direct(&managers[1]),
        )
        .users(
            GroupUsersRequest::default()
                .direct(&users[0])
                .direct(&users[1]),
        )
        .monitors(
            GroupUsersRequest::default()
                .direct(&monitors[0])
                .direct(&monitors[1]),
        );
    // remove the optional description
    group_req.description = None;
    // create the group
    let resp = client.groups.create(&group_req).await?;
    is!(resp.status().as_u16(), 204);
    // now try to recreate this group with a description
    group_req = group_req.description("This description should not be set on the existing group");
    let resp = client.groups.create(&group_req).await;
    // check that the creating an existing group returns a 401 Unauthorized
    fail!(resp, StatusCode::UNAUTHORIZED);
    // check that the description was not set on the existing group
    let group = client.groups.get(&group_req.name).await?;
    is!(group.description, None::<String>);
    Ok(())
}

#[tokio::test]
async fn get() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0);
    // get the group and make it matches the request
    let retrieved = client.groups.get(&group.name).await?;
    is!(retrieved, group);
    Ok(())
}

#[tokio::test]
async fn list() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create 20 random groups in Thorium
    let groups = generators::groups(20, &client).await?;
    // get the names of all the groups we have created
    let names: Vec<String> = groups.iter().map(|group| group.name.clone()).collect();
    // list the groups we just created
    let resp = client.groups.list().page(500).exec().await?;
    // make sure all the groups we tried to create are in our list
    for group in names {
        is_in!(resp.names, group);
    }
    Ok(())
}

#[tokio::test]
async fn list_details() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create 20 random groups in Thorium
    let group_reqs = generators::groups(20, &client).await?;
    // list the groups we just created
    let resp = client.groups.list().page(100).details().exec().await?;
    // make sure all the group details we tried to create are in our list
    vec_in_vec!(&group_reqs, &resp.details);
    Ok(())
}

#[tokio::test]
async fn delete() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create a random group then delete it
    let group = generators::gen_group();
    client.groups.create(&group).await?;
    // delete the group we just created
    let resp = client.groups.delete(&group.name).await?;
    is!(resp.status().as_u16(), 204);
    // make sure the delete worked by trying to recreate the group
    client.groups.create(&group).await?;
    // delete the group we just created
    let resp = client.groups.delete(&group.name).await?;
    is!(resp.status().as_u16(), 204);
    Ok(())
}

#[tokio::test]
async fn delete_recreate() -> Result<(), thorium::Error> {
    let client = test_utilities::admin_client().await?;
    // create a random group then delete it
    let group = generators::gen_group();
    client.groups.create(&group).await?;
    // delete the group we just created
    let resp = client.groups.delete(&group.name).await?;
    is!(resp.status().as_u16(), 204);
    // make sure the delete worked by trying to recreate the group
    client.groups.create(&group).await?;
    // delete the group we just created
    let resp = client.groups.delete(&group.name).await?;
    is!(resp.status().as_u16(), 204);
    // try to recreate our deleted group
    let resp = client.groups.create(&group).await?;
    is!(resp.status().as_u16(), 204);
    Ok(())
}

#[tokio::test]
async fn delete_deletes_network_policies() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create two groups
    let groups = generators::groups(2, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    // create many network policies in that group
    let _network_policies = generators::network_policies(&groups, 100, &client).await?;
    let deleted_group = &groups[1];
    // delete one of the groups we just created
    let resp = client.groups.delete(deleted_group).await?;
    is!(resp.status().as_u16(), 204);
    // make sure those network policies were deleted from that group
    let mut cursor = client
        .network_policies
        .list(&NetworkPolicyListOpts::default().groups(groups.clone()))
        .await?;
    loop {
        for policy in &cursor.data {
            // make sure the deleted group is not in any of the policies' groups
            is_not_in!(policy.groups, deleted_group);
        }
        if cursor.exhausted() {
            break;
        }
        cursor.refill().await?;
    }
    Ok(())
}

#[tokio::test]
async fn update() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // Create a group
    let group = generators::groups(1, &client).await?.remove(0).name;
    // Create the users to add to this group
    let owners = generators::users(2, &client).await?;
    let managers = generators::users(2, &client).await?;
    let users = generators::users(2, &client).await?;
    let monitors = generators::users(2, &client).await?;
    // build an update for the group we just created
    let update = GroupUpdate::default()
        .owners(
            GroupUsersUpdate::default()
                .direct_add(&owners[0])
                .direct_add(&owners[1]),
        )
        .managers(
            GroupUsersUpdate::default()
                .direct_add(&managers[0])
                .direct_add(&managers[1]),
        )
        .users(
            GroupUsersUpdate::default()
                .direct_add(&users[0])
                .direct_add(&users[1]),
        )
        .monitors(
            GroupUsersUpdate::default()
                .direct_add(&monitors[0])
                .direct_add(&monitors[1]),
        )
        .description("Updated description");
    // update the group and check the response code
    client.groups.update(&group, &update).await?;
    // get the group and make sure our updates were applied
    let updated = client.groups.get(&group).await?;
    is!(updated, update);
    // make sure our users all have this group now
    // this only checks the user roles but it should be accurate for all roles
    for user in &users {
        // get this user's data
        let user_data = client.users.get(user).await?;
        // make sure they are now members
        is_in!(user_data.groups, group);
    }
    // move the users to totally different groups to test role reassignment;
    // additionally test clearing optional values
    let update = GroupUpdate::default()
        .owners(
            GroupUsersUpdate::default()
                .direct_remove(&owners[0])
                .direct_remove(&owners[1])
                .direct_add(&managers[0])
                .direct_add(&managers[1]),
        )
        .managers(
            GroupUsersUpdate::default()
                .direct_remove(&managers[0])
                .direct_remove(&managers[1])
                .direct_add(&users[0])
                .direct_add(&users[1]),
        )
        .users(
            GroupUsersUpdate::default()
                .direct_remove(&users[0])
                .direct_remove(&users[1])
                .direct_add(&monitors[0])
                .direct_add(&monitors[1]),
        )
        .monitors(
            GroupUsersUpdate::default()
                .direct_remove(&monitors[0])
                .direct_remove(&monitors[1])
                .direct_add(&owners[0])
                .direct_add(&owners[1]),
        )
        .clear_description();
    // update the group and check the response code
    client.groups.update(&group, &update).await?;
    // get the group and make sure our updates were applied
    let updated = client.groups.get(&group).await?;
    is!(updated, update);
    Ok(())
}
