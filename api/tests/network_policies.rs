//! Tests network policies routes
//!
//! Because testing actual network policy application requires actually interacting
//! with the K8's API, this only tests that the Thorium network policy API route
//! functions properly, not that the policies themselves function.

use futures::stream::{self, StreamExt, TryStreamExt};
use itertools::Itertools;
use std::iter;
use uuid::Uuid;

use test_utilities::generators;
use thorium::models::{
    ImageNetworkPolicyUpdate, ImageUpdate, NetworkPolicyCustomK8sRule, NetworkPolicyCustomLabel,
    NetworkPolicyListOpts, NetworkPolicyRequest, NetworkPolicyRuleRaw, NetworkPolicyUpdate,
};
use thorium::test_utilities;
use thorium::{contains, fail, is, is_empty, iter_in_iter, vec_in_vec};

#[tokio::test]
async fn create() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // generate groups
    let groups = generators::groups(5, &client)
        .await?
        .into_iter()
        .map(|req| req.name)
        .collect::<Vec<String>>();
    // create a network policy request
    let req = generators::gen_network_policy(&groups);
    // test that we create the network policy successfully
    let resp = client.network_policies.create(req).await?;
    is!(resp.status().as_u16(), 204);
    Ok(())
}

#[tokio::test]
async fn create_bad() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // get user client
    let user_client = test_utilities::generators::client(&client).await?;
    // generate groups
    let groups = generators::groups(5, &client)
        .await?
        .into_iter()
        .map(|req| req.name)
        .collect::<Vec<String>>();
    // create a network policy request
    let mut req = generators::gen_network_policy(&groups);
    // fail to create a network policy as a non-admin
    let resp = user_client.network_policies.create(req.clone()).await;
    fail!(resp, 401);
    // fail to create a network policy with an empty name
    req.name = String::new();
    let resp = client.network_policies.create(req).await;
    fail!(resp, 400);
    // fail to create a network policy request with settings with empty groups
    let mut req = generators::gen_network_policy(&groups);
    req = req.add_egress_rule(NetworkPolicyRuleRaw::default().group(""));
    let resp = client.network_policies.create(req).await;
    fail!(resp, 400);
    // fail to create a network policy when a network policy with that name already exists in a group
    let req = generators::gen_network_policy(&groups);
    client.network_policies.create(req.clone()).await?;
    let mut already_exists_req = generators::gen_network_policy(&groups);
    already_exists_req.name = req.name;
    let resp = client
        .network_policies
        .create(already_exists_req.clone())
        .await;
    fail!(resp, 400, "already exists");
    // fail to create a network policy for groups that don't exist
    let mut req = generators::gen_network_policy(&groups);
    req.groups = vec!["no-exists".to_string()];
    let resp = client.network_policies.create(req).await;
    fail!(resp, 404, "groups doesn't exist");
    // fail to create a network policy with bad labels
    let mut req = generators::gen_network_policy(&groups);
    req = req.add_ingress_rule(NetworkPolicyRuleRaw::default().custom_rule(
        NetworkPolicyCustomK8sRule::new(
            Some(vec![NetworkPolicyCustomLabel::new(
                "THIS*#(IS@()NOT_(ALLOWED",
                "oops",
            )]),
            None,
        ),
    ));
    let resp = client.network_policies.create(req).await;
    fail!(resp, 400, "THIS*#(IS@()NOT_(ALLOWED");
    // fail to create a network policy with a bad cidr
    let mut req = generators::gen_network_policy(&groups);
    // here the cidr's network length is wrong
    req = req.add_ingress_rule(NetworkPolicyRuleRaw::default().ip_block("10.0.0.1/24", None));
    let resp = client.network_policies.create(req).await;
    fail!(resp, 400, "parse CIDR");
    Ok(())
}

#[tokio::test]
async fn get() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create groups
    let groups = generators::groups(2, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    // get user client
    let user_client = test_utilities::generators::client(&client).await?;
    // create a group for the user
    let user_group = generators::groups(1, &user_client).await?.remove(0).name;
    // create a network policy request
    let req = generators::gen_network_policy(&groups);
    // create the network policy
    client.network_policies.create(req.clone()).await?;
    // create an image with that network policy
    let image_req = generators::gen_image(&groups[0]).network_policy(&req.name);
    client.images.create(&image_req).await?;
    // get the network policy
    let network_policy = client.network_policies.get(&req.name, None).await?;
    // check that the network policy is the same as the request
    is!(network_policy, req);
    // check that the image is in the used_by list
    let used_by_images = network_policy.used_by.get(&groups[0]).unwrap();
    contains!(used_by_images, &image_req.name);
    // attempt to get that network policy as a user who is not in at least one of its groups
    let resp = user_client.network_policies.get(&req.name, None).await;
    // expect a 404
    fail!(resp, 404);
    // create a network policy in the user's group
    let groups_combined = groups
        .iter()
        .cloned()
        .chain(iter::once(user_group.clone()))
        .collect::<Vec<String>>();
    let mut req = generators::gen_network_policy(&groups_combined);
    client.network_policies.create(req.clone()).await?;
    // attempt to get that network policy as the user who is in one of the groups
    let network_policy = user_client.network_policies.get(&req.name, None).await?;
    // the retrieved network policy will not have the group the user is not in,
    // so remove all the groups that aren't the user's group
    req.groups.retain(|g| g == &user_group);
    for rule in req
        .ingress
        .iter_mut()
        .chain(req.egress.iter_mut())
        .flatten()
    {
        rule.allowed_groups.retain(|g| g == &user_group);
    }
    is!(network_policy, req);
    // get the network policy with its ID as well
    let network_policy = user_client
        .network_policies
        .get(&req.name, Some(network_policy.id))
        .await?;
    is!(network_policy, req);
    Ok(())
}

#[tokio::test]
async fn get_bad() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create groups
    let groups = generators::groups(2, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    // create a network policy request
    let req = generators::gen_network_policy(&[groups[0].clone()]);
    let mut req_2 = generators::gen_network_policy(&[groups[1].clone()]);
    req_2.name.clone_from(&req.name);
    // create different network policies with the same name
    client.network_policies.create(req.clone()).await?;
    client.network_policies.create(req_2.clone()).await?;
    // try to get the network policy with only its name
    let resp = client.network_policies.get(&req.name, None).await;
    fail!(resp, 400, "specify the network policy's ID");
    Ok(())
}

#[tokio::test]
async fn update() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create groups
    let mut groups = generators::groups(2, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    // create a network policy request
    let req = generators::gen_network_policy(&groups)
        // ensure that the policy has rules
        .add_ingress_rule(generators::gen_network_policy_rule(&groups))
        .add_egress_rule(generators::gen_network_policy_rule(&groups));
    // create the network policy
    client.network_policies.create(req.clone()).await?;
    // create an image with that network policy
    let image_req = generators::gen_image(&groups[0]).network_policy(&req.name);
    client.images.create(&image_req).await?;
    // get the created network policy
    let network_policy = client.network_policies.get(&req.name, None).await?;
    // update the network policy
    let remove_group = groups.pop().unwrap();
    let new_group = generators::groups(2, &client).await?.remove(0).name;
    let new_name = "cool-new-name";
    let update = NetworkPolicyUpdate::default()
        .new_name(new_name)
        .remove_egress_rule(
            network_policy
                .egress
                .expect("netpol must have egress rules")
                .last()
                .expect("netpol must have a last egress rule")
                .id,
        )
        .add_ingress_rule(generators::gen_network_policy_rule(&groups))
        .add_group(new_group)
        .remove_group(&remove_group)
        .forced_policy(true)
        .default_policy(true);
    client
        .network_policies
        .update(&req.name, None, &update)
        .await?;
    // get the updated network policy
    let network_policy = client
        .network_policies
        .get(new_name, Some(network_policy.id))
        .await?;
    // make sure the network policy is updated
    is!(network_policy, update);
    // make sure the image's network policies were updated
    let image = client.images.get(&groups[0], &image_req.name).await?;
    contains!(image.network_policies, new_name);
    // remove the network policy from the image
    let image_update = ImageUpdate::default()
        .network_policies(ImageNetworkPolicyUpdate::default().remove_policy(new_name));
    client
        .images
        .update(&groups[0], &image_req.name, &image_update)
        .await?;
    // update the network policy now that it's not attached to any image
    let new_name = "cool-new-name2";
    let update = NetworkPolicyUpdate::default()
        .new_name(new_name)
        // also clear all rules and set to deny all
        .deny_all_ingress()
        .deny_all_egress();
    client
        .network_policies
        .update(&network_policy.name, Some(network_policy.id), &update)
        .await?;
    // get the updated network policy
    let network_policy = client
        .network_policies
        .get(new_name, Some(network_policy.id))
        .await?;
    // make sure the network policy is updated
    is!(network_policy, update);
    // add it back to the image and update one more time
    let image_update = ImageUpdate::default()
        .network_policies(ImageNetworkPolicyUpdate::default().add_policy(new_name));
    client
        .images
        .update(&groups[0], &image_req.name, &image_update)
        .await?;
    // update one more time
    let new_name = "cool-new-name3";
    let update = NetworkPolicyUpdate::default()
        .new_name(new_name)
        // allow all ingress/egress as well
        .clear_ingress()
        .clear_egress();
    client
        .network_policies
        .update(&network_policy.name, Some(network_policy.id), &update)
        .await?;
    // get the updated network policy
    let network_policy = client
        .network_policies
        .get(new_name, Some(network_policy.id))
        .await?;
    // make sure the network policy is updated
    is!(network_policy, update);
    Ok(())
}

#[tokio::test]
async fn update_bad() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create groups
    let mut groups = generators::groups(2, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    // get user client
    let user_client = test_utilities::generators::client(&client).await?;
    // create a group for the user and add it to our list
    let user_group = generators::groups(1, &user_client).await?.remove(0).name;
    groups.push(user_group);
    // create a network policy request
    let req = generators::gen_network_policy(&groups);
    // create the network policy
    client.network_policies.create(req.clone()).await?;
    // create an image with that network policy
    let image_req = generators::gen_image(&groups[0]).network_policy(&req.name);
    client.images.create(&image_req).await?;
    // try to update with an empty update
    let empty_update = NetworkPolicyUpdate::default();
    let resp = client
        .network_policies
        .update(&req.name, None, &empty_update)
        .await;
    fail!(resp, 400, "is empty");
    // try to update a network policy as a user
    let update = NetworkPolicyUpdate::default().forced_policy(true);
    let resp = user_client
        .network_policies
        .update(&req.name, None, &update)
        .await;
    fail!(resp, 401);
    // try to add a group to the network policy that it already has
    let update = NetworkPolicyUpdate::default().add_group(&groups[0]);
    let resp = client
        .network_policies
        .update(&req.name, None, &update)
        .await;
    fail!(resp, 400, "already contains group");
    // try to remove groups from the network policy that it doesn't have
    let update = NetworkPolicyUpdate::default().remove_group("nonexistent-group");
    let resp = client
        .network_policies
        .update(&req.name, None, &update)
        .await;
    fail!(resp, 400, "does not contain group");
    // try to remove all of the network policy's groups
    let update = NetworkPolicyUpdate::default().remove_groups(&groups);
    let resp = client
        .network_policies
        .update(&req.name, None, &update)
        .await;
    fail!(resp, 400, "delete all of a network policy's groups");
    // try to add a group to the network policy that doesn't exist
    let update = NetworkPolicyUpdate::default().add_group("nonexistent-group");
    let resp = client
        .network_policies
        .update(&req.name, None, &update)
        .await;
    fail!(resp, 404, "groups to add doesn't exist");
    // try to add bad rules to the network policy
    let update =
        NetworkPolicyUpdate::default().add_ingress_rule(NetworkPolicyRuleRaw::default().group(""));
    let resp = client
        .network_policies
        .update(&req.name, None, &update)
        .await;
    fail!(resp, 400, "one or more groups with empty names");
    // try to remove a rule from the network policy that it doesn't have
    let update = NetworkPolicyUpdate::default().remove_egress_rule(Uuid::new_v4());
    let resp = client
        .network_policies
        .update(&req.name, None, &update)
        .await;
    fail!(resp, 404, "egress rules to be removed is not found");
    // try to update with an empty name
    let update = NetworkPolicyUpdate::default().new_name("");
    let resp = client
        .network_policies
        .update(&req.name, None, &update)
        .await;
    fail!(resp, 400, "empty name");
    // try to update by allowing/denying all ingress/egress
    let update = NetworkPolicyUpdate::default()
        .clear_ingress()
        .deny_all_ingress();
    let resp = client
        .network_policies
        .update(&req.name, None, &update)
        .await;
    fail!(resp, 400, "both clear ingress rules and deny all ingress");
    let update = NetworkPolicyUpdate::default()
        .clear_egress()
        .deny_all_egress();
    let resp = client
        .network_policies
        .update(&req.name, None, &update)
        .await;
    fail!(resp, 400, "both clear egress rules and deny all egress");
    Ok(())
}

#[tokio::test]
async fn delete() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create groups
    let groups = generators::groups(2, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    // create a network policy request
    let req = generators::gen_network_policy(&groups);
    // create the network policy
    client.network_policies.create(req.clone()).await?;
    // create an image with that network policy
    let image_req = generators::gen_image(&groups[0]).network_policy(&req.name);
    client.images.create(&image_req).await?;
    // delete the network policy
    client.network_policies.delete(&req.name, None).await?;
    // make sure the network policy is gone
    let resp = client.network_policies.get(&req.name, None).await;
    fail!(resp, 404);
    // make sure the image no longer has that network policy
    let image = client.images.get(&groups[0], &image_req.name).await?;
    is_empty!(image.network_policies);
    // create a network policy request
    let req = generators::gen_network_policy(&groups);
    // create the network policy
    client.network_policies.create(req.clone()).await?;
    // get the network policy (to get its ID)
    let policy = client.network_policies.get(&req.name, None).await?;
    // delete the policy with its name AND ID
    client
        .network_policies
        .delete(&req.name, Some(policy.id))
        .await?;
    // make sure the network policy is gone
    let resp = client
        .network_policies
        .get(&req.name, Some(policy.id))
        .await;
    fail!(resp, 404);
    Ok(())
}

#[tokio::test]
async fn delete_bad() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create groups
    let mut groups = generators::groups(2, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    // get user client
    let user_client = test_utilities::generators::client(&client).await?;
    // create a group for the user and add it to our list
    let user_group = generators::groups(1, &user_client).await?.remove(0).name;
    groups.push(user_group);
    // create a network policy request
    let req = generators::gen_network_policy(&groups);
    // create the network policy
    client.network_policies.create(req.clone()).await?;
    // try to delete the network policy as a user and expect an unauthorized error
    let resp = user_client.network_policies.delete(&req.name, None).await;
    fail!(resp, 401);
    Ok(())
}

#[tokio::test]
async fn get_all_default() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    let group = generators::groups(1, &client).await?.remove(0).name;
    let num_default_policies = 50;
    let num_non_default_policies = 100;
    let default_policy_reqs: Vec<NetworkPolicyRequest> = (0..num_default_policies)
        .map(|_| {
            let mut req = generators::gen_network_policy(&[group.clone()]);
            req.default_policy = true;
            req
        })
        .collect();
    let non_default_policy_reqs: Vec<NetworkPolicyRequest> = (0..num_non_default_policies)
        .map(|_| {
            let mut req = generators::gen_network_policy(&[group.clone()]);
            req.default_policy = false;
            req
        })
        .collect();
    // create all of the policies concurrently
    stream::iter(
        default_policy_reqs
            .iter()
            .cloned()
            .chain(non_default_policy_reqs.into_iter()),
    )
    .map(Ok::<NetworkPolicyRequest, thorium::Error>)
    .try_for_each_concurrent(50, |req| {
        let client_ref = &client;
        async move {
            client_ref.network_policies.create(req).await?;
            Ok(())
        }
    })
    .await?;
    // list all of the default policies
    let default_policies = client.network_policies.get_all_default(&group).await?;
    // make sure we got the correct number back
    is!(
        default_policies.len(),
        num_default_policies,
        "correct num default"
    );
    let names_sorted: Vec<String> = default_policies
        .into_iter()
        .map(|line| line.name)
        .sorted()
        .collect();
    let names_reqs_sorted: Vec<String> = default_policy_reqs
        .into_iter()
        .map(|req| req.name)
        .sorted()
        .collect();
    is!(
        names_sorted,
        names_reqs_sorted,
        "reqs and retrieved identical"
    );
    Ok(())
}

#[tokio::test]
async fn list_names() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    let groups = generators::groups(10, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    // generate a ton of policies
    let num_policies = 100;
    let policy_reqs = generators::network_policies(&groups, num_policies, &client).await?;
    // set limit higher to check that the cursor doesn't get stuck
    let opts = NetworkPolicyListOpts::default()
        .groups(&groups)
        .limit(num_policies * 2);
    // list all policies
    let mut cursor = client.network_policies.list(&opts).await?;
    let mut policies = Vec::new();
    loop {
        for policy_line in cursor.data.drain(..) {
            policies.push(policy_line);
        }
        if cursor.exhausted() {
            break;
        }
        cursor.refill().await?;
    }
    // make sure we got all policies
    is!(policies.len(), num_policies, "correct policy number");
    iter_in_iter!(
        policy_reqs.iter().map(|p| &p.name),
        policies.iter().map(|p| &p.name),
        "listed all policies"
    );
    // make sure the policies are sorted by name
    let mut policies_sorted = policies.clone();
    policies_sorted.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    is!(policies, policies_sorted, "policies list sorted by name");
    Ok(())
}

#[tokio::test]
async fn list_details() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    let groups = generators::groups(10, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    // generate a ton of policies
    let num_policies = 100;
    let policy_reqs = generators::network_policies(&groups, num_policies, &client).await?;
    // set limit higher to check that the cursor doesn't get stuck
    let opts = NetworkPolicyListOpts::default()
        .groups(&groups)
        .limit(num_policies * 2);
    // list all policies
    let mut cursor = client.network_policies.list_details(&opts).await?;
    let mut policies = Vec::new();
    loop {
        for policy_line in cursor.data.drain(..) {
            policies.push(policy_line);
        }
        if cursor.exhausted() {
            break;
        }
        cursor.refill().await?;
    }
    // make sure we got all policies
    is!(policies.len(), num_policies, "correct policy number");
    vec_in_vec!(policies, policy_reqs, "listed all policies details");
    // make sure the policies are sorted by name
    let mut policies_sorted = policies.clone();
    policies_sorted.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    is!(policies, policies_sorted, "policies details sorted by name");
    // make sure that if we get a partial list, we get ALL of the policies' data (groups)
    let groups = generators::groups(10, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    let num_policies = 5;
    let num_policies_req = 3;
    let policy_reqs = generators::network_policies(&groups, num_policies, &client).await?;
    let opts = NetworkPolicyListOpts::default()
        .groups(&groups)
        .limit(num_policies_req);
    // list only some of the policies
    let mut cursor = client.network_policies.list_details(&opts).await?;
    let mut policies = Vec::new();
    loop {
        for policy_line in cursor.data.drain(..) {
            policies.push(policy_line);
        }
        if cursor.exhausted() {
            break;
        }
        cursor.refill().await?;
    }
    // make sure we got all policies
    is!(
        policies.len(),
        num_policies_req,
        "correct policy partial request number"
    );
    vec_in_vec!(policies, policy_reqs, "correct policy partial request");
    Ok(())
}

#[tokio::test]
async fn list_details_from_partial_group() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    let groups = generators::groups(10, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    // generate policies
    let num_policies = 10;
    let policy_reqs = generators::network_policies(&groups, num_policies, &client).await?;
    // make sure when the user lists network policies we don't get any of the groups the user can't see
    let opts = NetworkPolicyListOpts::default()
        .groups([groups.first().unwrap().clone()])
        .limit(num_policies)
        .page_size(num_policies / 2);
    let mut cursor = client.network_policies.list_details(&opts).await?;
    let mut policies = Vec::new();
    loop {
        for policy in cursor.data.drain(..) {
            policies.push(policy);
        }
        if cursor.exhausted() {
            break;
        }
        cursor.refill().await?;
    }
    // make sure we got all policies and the details only have the user's group
    is!(policies.len(), num_policies, "got all policies");
    vec_in_vec!(policies, policy_reqs, "policies have all details");
    Ok(())
}

#[tokio::test]
async fn list_details_user() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    let mut groups = generators::groups(10, &client)
        .await?
        .into_iter()
        .map(|g| g.name)
        .collect::<Vec<String>>();
    // get user client
    let user_client = generators::client(&client).await?;
    // create a group for the user and add it to the list
    let user_group = generators::groups(1, &user_client).await?.remove(0).name;
    groups.push(user_group.clone());
    // generate policies
    let num_policies = 10;
    let _ = generators::network_policies(&groups, num_policies, &client).await?;
    // make sure when the user lists network policies we don't get any of the groups the user can't see
    let opts = NetworkPolicyListOpts::default()
        .groups([user_group.clone()])
        .limit(num_policies)
        .page_size(num_policies / 2);
    let mut cursor = user_client.network_policies.list_details(&opts).await?;
    let mut policies = Vec::new();
    loop {
        for policy in cursor.data.drain(..) {
            policies.push(policy);
        }
        if cursor.exhausted() {
            break;
        }
        cursor.refill().await?;
    }
    // make sure we got all policies and the details only have the user's group
    is!(policies.len(), num_policies, "got all policies");
    let policy_groups = &policies.first().unwrap().groups;
    is!(policy_groups.len(), 1, "only user's group");
    is!(
        policy_groups.first().unwrap(),
        &user_group,
        "group is user's group"
    );
    // make sure none of the allowed groups have more than one group (the user's group)
    // the user shouldn't be able to see the other groups are allowed; otherwise, those groups'
    // existence would be leaked to the user
    for policy in &policies {
        for rule in policy.ingress.iter().flatten() {
            is!(
                (rule.allowed_groups.len() <= 1),
                true,
                "ingress allowed groups only user's"
            );
        }
        for rule in policy.egress.iter().flatten() {
            is!(
                (rule.allowed_groups.len() <= 1),
                true,
                "egress allowed groups only user's"
            );
        }
    }
    Ok(())
}

#[tokio::test]
async fn used_by() -> Result<(), thorium::Error> {
    // get admin client
    let client = test_utilities::admin_client().await?;
    // create groups
    let group = generators::groups(1, &client).await?.remove(0).name;
    // get user client
    let user_client = test_utilities::generators::client(&client).await?;
    // create a group for the user
    let user_group = generators::groups(1, &user_client).await?.remove(0).name;
    // create a network policy request in both groups
    let groups = &[group.clone(), user_group.clone()];
    let req = generators::gen_network_policy(groups);
    // create the network policy
    client.network_policies.create(req.clone()).await?;
    // create two images with that network policy in different groups
    let image_req = generators::gen_image(&group).network_policy(&req.name);
    client.images.create(&image_req).await?;
    let image_req_user = generators::gen_image(&user_group).network_policy(&req.name);
    client.images.create(&image_req_user).await?;
    // get the network policy as admin
    let network_policy = client.network_policies.get(&req.name, None).await?;
    // check that the image is in the used_by list
    let used_by_images = network_policy.used_by.get(&group).unwrap();
    contains!(used_by_images, &image_req.name);
    let used_by_images_user = network_policy.used_by.get(&user_group).unwrap();
    contains!(used_by_images_user, &image_req_user.name);
    // get the network policy as user
    let network_policy = user_client.network_policies.get(&req.name, None).await?;
    // make sure that user can only see used by in their own groups
    let used_by_images_user = network_policy.used_by.get(&user_group).unwrap();
    contains!(used_by_images_user, &image_req_user.name);
    is!(
        network_policy.used_by.get(&group),
        None::<&Vec<String>>,
        "no admin group in `used by`"
    );
    // remove the network policy from the second image
    let image_update = ImageUpdate::default()
        .network_policies(ImageNetworkPolicyUpdate::default().remove_policy(&network_policy.name));
    client
        .images
        .update(&group, &image_req.name, &image_update)
        .await?;
    // check that the image was removed from the used_by list
    let mut network_policy = client.network_policies.get(&req.name, None).await?;
    is!(
        network_policy.used_by.remove(&group).unwrap_or_default(),
        Vec::<String>::default(),
        "admin image removed"
    );
    let used_by_images_user = network_policy.used_by.get(&user_group).unwrap();
    contains!(used_by_images_user, &image_req_user.name);
    Ok(())
}
