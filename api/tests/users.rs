//! Tests the users routes in Thorium

use chrono::{Duration, Utc};
use thorium::models::{GroupRequest, ScopedTokenRequest, ScopedTokenUpdate, UserRole, UserUpdate};
use thorium::test_utilities::{self, generators};
use thorium::{Error, Thorium, fail, is, is_not};
use uuid::Uuid;

#[tokio::test]
async fn delete() -> Result<(), Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // get a user client
    let client = generators::client(&client).await?;
    // get our users info
    let info = client.users.info().await?;
    // delete our user
    client.users.delete(&info.username).await?;
    Ok(())
}

/// Create a group owned by the given clients user
///
/// # Arguments
///
/// * `client` - The client to create a group with
async fn create_group(client: &Thorium) -> Result<String, Error> {
    // generate a random lowercase group name
    let name = Uuid::new_v4().simple().to_string();
    // build a group request for this group
    let req = GroupRequest::new(&name);
    // create this group
    client.groups.create(&req).await?;
    Ok(name)
}

/// Build a client that authenticates with a scoped token
///
/// # Arguments
///
/// * `client` - The client to get a host from
/// * `token` - The scoped token value to authenticate with
async fn scoped_client(client: &Thorium, token: &str) -> Result<Thorium, Error> {
    // build a client that uses this scoped token
    Thorium::build(client.host.clone())
        .token(token)
        .build()
        .await
}

#[tokio::test]
async fn scoped_token_crud() -> Result<(), Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // get a user client
    let client = generators::client(&admin).await?;
    // create two groups owned by this user
    let in_scope = create_group(&client).await?;
    let out_of_scope = create_group(&client).await?;
    // build a scoped token request limited to one group
    let req = ScopedTokenRequest::new("crud-token").group(&in_scope);
    // create our scoped token
    let scoped = client.users.create_scoped_token(&req).await?;
    // make sure our scoped token has the right settings
    is!(scoped.name, "crud-token");
    is!(scoped.groups, vec![in_scope.clone()]);
    // build a client that authenticates with our scoped token
    let scoped_thorium = scoped_client(&client, &scoped.token).await?;
    // make sure whoami shows only in scope groups and the scoped tokens value
    let info = scoped_thorium.users.info().await?;
    is!(info.groups, vec![in_scope.clone()]);
    is!(info.token, scoped.token);
    // make sure we can get our in scope group
    scoped_thorium.groups.get(&in_scope).await?;
    // but not our out of scope group
    fail!(scoped_thorium.groups.get(&out_of_scope).await, 401);
    // make sure our primary token can still see both groups
    client.groups.get(&in_scope).await?;
    client.groups.get(&out_of_scope).await?;
    // make sure our scoped token shows up when listing
    let listed = client.users.list_scoped_tokens().await?;
    is!(listed.len(), 1);
    // and when getting it by name
    let got = client.users.get_scoped_token("crud-token").await?;
    is!(got.token, scoped.token);
    // delete our scoped token
    client.users.delete_scoped_token("crud-token").await?;
    // make sure our scoped token no longer works
    fail!(scoped_thorium.users.info().await, 401);
    // and is no longer listed
    let listed = client.users.list_scoped_tokens().await?;
    is!(listed.len(), 0);
    Ok(())
}

#[tokio::test]
async fn scoped_token_create_invalid() -> Result<(), Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // get a user client
    let client = generators::client(&admin).await?;
    // create a group owned by this user
    let group = create_group(&client).await?;
    // make sure groups outside our own are rejected
    let req = ScopedTokenRequest::new("invalid-groups").group("not-our-group");
    fail!(client.users.create_scoped_token(&req).await, 401);
    // make sure empty scopes are rejected
    let req = ScopedTokenRequest::new("empty-scope");
    fail!(client.users.create_scoped_token(&req).await, 400);
    // make sure expirations in the past are rejected
    let req = ScopedTokenRequest::new("past-expires")
        .group(&group)
        .expires(Utc::now() - Duration::days(1));
    fail!(client.users.create_scoped_token(&req).await, 400);
    // make sure invalid names are rejected
    let req = ScopedTokenRequest::new("NotLowercase").group(&group);
    fail!(client.users.create_scoped_token(&req).await, 400);
    // make sure duplicate names are rejected
    let req = ScopedTokenRequest::new("duplicate").group(&group);
    client.users.create_scoped_token(&req).await?;
    fail!(client.users.create_scoped_token(&req).await, 409);
    Ok(())
}

#[tokio::test]
async fn scoped_token_update() -> Result<(), Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // get a user client
    let client = generators::client(&admin).await?;
    // create two groups owned by this user
    let group_a = create_group(&client).await?;
    let group_b = create_group(&client).await?;
    // create a scoped token limited to our first group
    let req = ScopedTokenRequest::new("update-token").group(&group_a);
    let scoped = client.users.create_scoped_token(&req).await?;
    // build a client that authenticates with our scoped token
    let scoped_thorium = scoped_client(&client, &scoped.token).await?;
    // add our second group to this scoped tokens scope
    let update = ScopedTokenUpdate::default().add_group(&group_b);
    let updated = client
        .users
        .update_scoped_token("update-token", &update)
        .await?;
    // make sure our scope now contains both groups
    is!(updated.groups.len(), 2);
    is!(updated.groups.contains(&group_b), true);
    // make sure our scoped tokens value was not changed
    is!(updated.token, scoped.token);
    // make sure our scoped token still authenticates after the update
    let info = scoped_thorium.users.info().await?;
    is!(info.groups.len(), 2);
    // and can now access our second group
    scoped_thorium.groups.get(&group_b).await?;
    // remove our first group from this scoped tokens scope
    let update = ScopedTokenUpdate::default().remove_group(&group_a);
    let updated = client
        .users
        .update_scoped_token("update-token", &update)
        .await?;
    // make sure our scope only contains our second group
    is!(updated.groups, vec![group_b.clone()]);
    // and our first group is no longer accessible with our scoped token
    fail!(scoped_thorium.groups.get(&group_a).await, 401);
    // set an expiration date for this scoped token
    let expires = Utc::now() + Duration::days(1);
    let update = ScopedTokenUpdate::default().expires(expires);
    let updated = client
        .users
        .update_scoped_token("update-token", &update)
        .await?;
    // make sure our expiration date was set
    is!(updated.expires, Some(expires));
    // clear this scoped tokens expiration date
    let update = ScopedTokenUpdate::default().clear_expires();
    let updated = client
        .users
        .update_scoped_token("update-token", &update)
        .await?;
    // make sure our expiration date was cleared
    is!(updated.expires.is_none(), true);
    Ok(())
}

#[tokio::test]
async fn scoped_token_update_invalid() -> Result<(), Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // get a user client
    let client = generators::client(&admin).await?;
    // create a group owned by this user
    let group = create_group(&client).await?;
    // create a scoped token limited to our group
    let req = ScopedTokenRequest::new("update-invalid").group(&group);
    client.users.create_scoped_token(&req).await?;
    // make sure updates to unknown scoped tokens are rejected
    let update = ScopedTokenUpdate::default().add_group(&group);
    fail!(
        client.users.update_scoped_token("unknown", &update).await,
        404
    );
    // make sure adding groups outside our own is rejected
    let update = ScopedTokenUpdate::default().add_group("not-our-group");
    fail!(
        client
            .users
            .update_scoped_token("update-invalid", &update)
            .await,
        401
    );
    // make sure removing all groups is rejected
    let update = ScopedTokenUpdate::default().remove_group(&group);
    fail!(
        client
            .users
            .update_scoped_token("update-invalid", &update)
            .await,
        400
    );
    // make sure expirations in the past are rejected
    let update = ScopedTokenUpdate::default().expires(Utc::now() - Duration::days(1));
    fail!(
        client
            .users
            .update_scoped_token("update-invalid", &update)
            .await,
        400
    );
    // make sure setting and clearing an expiration at once is rejected
    let update = ScopedTokenUpdate::default()
        .expires(Utc::now() + Duration::days(1))
        .clear_expires();
    fail!(
        client
            .users
            .update_scoped_token("update-invalid", &update)
            .await,
        400
    );
    // make sure empty updates are rejected
    let update = ScopedTokenUpdate::default();
    fail!(
        client
            .users
            .update_scoped_token("update-invalid", &update)
            .await,
        400
    );
    Ok(())
}

#[tokio::test]
async fn scoped_token_restricted_routes() -> Result<(), Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // get a user client
    let client = generators::client(&admin).await?;
    // create a group owned by this user
    let group = create_group(&client).await?;
    // create a scoped token for this user
    let req = ScopedTokenRequest::new("restricted").group(&group);
    let scoped = client.users.create_scoped_token(&req).await?;
    // build a client that authenticates with our scoped token
    let scoped_thorium = scoped_client(&client, &scoped.token).await?;
    // get our users info
    let info = client.users.info().await?;
    // make sure scoped tokens cannot manage scoped tokens
    let sneaky = ScopedTokenRequest::new("sneaky").group(&group);
    fail!(scoped_thorium.users.create_scoped_token(&sneaky).await, 401);
    fail!(scoped_thorium.users.list_scoped_tokens().await, 401);
    fail!(
        scoped_thorium.users.get_scoped_token("restricted").await,
        401
    );
    let sneaky_update = ScopedTokenUpdate::default().add_group(&group);
    fail!(
        scoped_thorium
            .users
            .update_scoped_token("restricted", &sneaky_update)
            .await,
        401
    );
    fail!(
        scoped_thorium.users.delete_scoped_token("restricted").await,
        401
    );
    // make sure scoped tokens cannot update users
    let update = UserUpdate {
        password: Some(generators::gen_string(24)),
        email: None,
        role: None,
        settings: None,
    };
    let resp = scoped_thorium.users.update(&info.username, update).await;
    fail!(resp, 401);
    // make sure scoped tokens cannot delete users
    fail!(scoped_thorium.users.delete(&info.username).await, 401);
    Ok(())
}

#[tokio::test]
async fn scoped_token_logout() -> Result<(), Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // get a user client
    let client = generators::client(&admin).await?;
    // create a group owned by this user
    let group = create_group(&client).await?;
    // create a scoped token for this user
    let req = ScopedTokenRequest::new("logout-token").group(&group);
    let scoped = client.users.create_scoped_token(&req).await?;
    // build a client that authenticates with our scoped token
    let scoped_thorium = scoped_client(&client, &scoped.token).await?;
    // make sure our scoped client works
    scoped_thorium.users.info().await?;
    // logout with our scoped token
    scoped_thorium.users.logout().await?;
    // make sure our old scoped token value no longer works
    fail!(scoped_thorium.users.info().await, 401);
    // but our primary token still works
    client.users.info().await?;
    // get the rotated value with our primary token
    let rotated = client.users.get_scoped_token("logout-token").await?;
    is_not!(rotated.token, scoped.token);
    // make sure the rotated value works
    let rotated_thorium = scoped_client(&client, &rotated.token).await?;
    rotated_thorium.users.info().await?;
    Ok(())
}

#[tokio::test]
async fn scoped_token_admin_demoted() -> Result<(), Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // create a group our admin owns
    let group = create_group(&admin).await?;
    // create another user to get info on
    let users = generators::users(1, &admin).await?;
    // make sure admins can get info on other users
    admin.users.get(&users[0]).await?;
    // create a scoped token as our admin
    let req = ScopedTokenRequest::new("demoted").group(&group);
    let scoped = admin.users.create_scoped_token(&req).await?;
    // build a client that authenticates with our scoped token
    let scoped_thorium = scoped_client(&admin, &scoped.token).await?;
    // make sure our scoped token was demoted to the user role
    let info = scoped_thorium.users.info().await?;
    is!(info.role, UserRole::User);
    is!(info.groups, vec![group.clone()]);
    // make sure our scoped token can no longer get info on other users
    fail!(scoped_thorium.users.get(&users[0]).await, 401);
    // but can still access data in the groups its scoped to
    scoped_thorium.groups.get(&group).await?;
    // clean up our admins scoped token
    admin.users.delete_scoped_token("demoted").await?;
    Ok(())
}

#[tokio::test]
async fn scoped_token_ephemeral() -> Result<(), Error> {
    // get admin client
    let admin = test_utilities::admin_client().await?;
    // get a user client
    let client = generators::client(&admin).await?;
    // create a group owned by this user
    let group = create_group(&client).await?;
    // create a scoped token that permanently expires almost immediately
    let req = ScopedTokenRequest::new("ephemeral")
        .group(&group)
        .expires(Utc::now() + Duration::seconds(2));
    let scoped = client.users.create_scoped_token(&req).await?;
    // build a client that authenticates with our scoped token
    let scoped_thorium = scoped_client(&client, &scoped.token).await?;
    // make sure our scoped token works before it expires
    scoped_thorium.users.info().await?;
    // wait for our scoped token to permanently expire
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    // make sure our scoped token no longer works
    fail!(scoped_thorium.users.info().await, 401);
    // and is no longer listed
    let listed = client.users.list_scoped_tokens().await?;
    is!(listed.len(), 0);
    Ok(())
}
