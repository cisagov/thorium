//! Handles network policies commands

use std::path::Path;
use thorium::{
    conf::BaseNetworkPolicy,
    models::{NetworkPolicyListLine, NetworkPolicyRequest, NetworkPolicyUpdate},
    CtlConf, Error, Thorium,
};

use crate::args::{Args, DescribeCommand, SearchParameterized};
use crate::utils;
use crate::{
    args::network_policies::{
        CreateNetworkPolicy, DefaultNetworkPolicies, DeleteNetworkPolicy, DescribeNetworkPolicies,
        GetNetworkPolicies, NetworkPolicies, NetworkPolicyFileFormat, NetworkPolicyRules,
        UpdateNetworkPolicy, VerifyNetworkPolicies,
    },
    err_not_admin,
};

struct NetworkPolicyLine;

impl NetworkPolicyLine {
    fn header() {
        println!("{:<64} | {:<36}", "POLICY NAME", "ID");
        println!("{:-<65}+{:-<37}", "", "");
    }

    fn print(network_policy_line: &NetworkPolicyListLine) {
        println!(
            "{:<64} | {:<37}",
            network_policy_line.name, network_policy_line.id
        );
    }
}

/// Get a list of network policies
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The get network policies command that was run
async fn get(thorium: Thorium, cmd: &GetNetworkPolicies) -> Result<(), Error> {
    let opts = cmd.build_network_policy_opts();
    let mut cursor = thorium.network_policies.list(&opts).await?;
    NetworkPolicyLine::header();
    loop {
        for network_policy_line in cursor.data.drain(..) {
            NetworkPolicyLine::print(&network_policy_line);
        }
        if cursor.exhausted() {
            break;
        }
        cursor.refill().await?;
    }
    Ok(())
}

/// Get a list of default network policies for a given group
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The default network policies command that was run
async fn get_default(thorium: Thorium, cmd: &DefaultNetworkPolicies) -> Result<(), Error> {
    let default_policies = thorium.network_policies.get_all_default(&cmd.group).await?;
    NetworkPolicyLine::header();
    for policy_list_line in default_policies {
        NetworkPolicyLine::print(&policy_list_line);
    }
    Ok(())
}

/// Describe network policies by displaying/saving their JSON-formatted details
///
/// * `thorium` - The Thorium client
/// * `cmd` - The [`DescribeNetworkPolicies`] command to execute
async fn describe(thorium: Thorium, cmd: &DescribeNetworkPolicies) -> Result<(), Error> {
    cmd.describe(&thorium).await
}

/// Create a network policy
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf`- The Thorctl config
/// * `cmd` - The create network policy command that was run
async fn create(thorium: Thorium, conf: CtlConf, cmd: &CreateNetworkPolicy) -> Result<(), Error> {
    // make a create request from the command
    let req = cmd.to_req(&conf)?;
    // send the request
    err_not_admin!(
        thorium.network_policies.create(req).await,
        "create network policies"
    );
    Ok(())
}

/// Delete a network policy
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `cmd` - The delete network policy command that was run
async fn delete(thorium: Thorium, cmd: &DeleteNetworkPolicy) -> Result<(), Error> {
    // send the request
    err_not_admin!(
        thorium.network_policies.delete(&cmd.name, cmd.id).await,
        "delete network policies"
    );
    Ok(())
}

/// Update a network policy
///
/// # Arguments
///
/// * `thorium` - The Thorium client
/// * `conf` - The Thorctl config
/// * `cmd` - The update network policy command that was run
async fn update(thorium: Thorium, conf: CtlConf, cmd: &UpdateNetworkPolicy) -> Result<(), Error> {
    // make a update request from the command
    let update = cmd.to_update(&conf)?;
    // send the request
    err_not_admin!(
        thorium
            .network_policies
            .update(&cmd.name, cmd.id, &update)
            .await,
        "update network policies"
    );
    Ok(())
}

/// Parse the rules for ingress and egress from a rules file
///
/// # Arguments
///
/// * `path` - The path to the rules file
/// * `format` - The format the rules file is in
/// * `api_url` - The URL to the API to use to provide a link to formatting information
///               on error
fn parse_rules_file(
    path: &Path,
    format: &NetworkPolicyFileFormat,
    api_url: &str,
) -> Result<NetworkPolicyRules, Error> {
    // open the rules file
    let file = std::fs::File::open(path).map_err(|err| {
        Error::new(format!(
            "Failed to open rules file at '{}': {}",
            path.to_string_lossy(),
            err
        ))
    })?;
    // try to deserialize the rules file
    let rules: NetworkPolicyRules = match format {
            NetworkPolicyFileFormat::Json => serde_json::from_reader(&file).map_err(|err| {
                Error::new(format!(
                    "Failed to deserialize JSON rules file at '{}': {}\n\n\
                    For formatting guidance on rules files, view the docs at '{}/docs/user/admins/network_policies.html#the-rules-file'",
                    path.to_string_lossy(),
                    err,
                    api_url
                ))
            })?,
            NetworkPolicyFileFormat::Yaml => serde_yaml::from_reader(&file).map_err(|err| {
                Error::new(format!(
                    "Failed to deserialize YAML rules file at '{}': {}\n\n\
                    For formatting guidance on rules files, view the docs at '{}/docs/user/admins/network_policies.html#the-rules-file'",
                    path.to_string_lossy(),
                    err,
                    api_url
                ))
            })?,
        };
    Ok(rules)
}

impl CreateNetworkPolicy {
    /// Generate a [`NetworkPolicyRequest`] from the options set in a
    /// `CreateNetworkPolicy` command
    ///
    /// # Arguments
    ///
    /// * `conf` - The Thorctl config used for Thoradm
    fn to_req(&self, conf: &CtlConf) -> Result<NetworkPolicyRequest, Error> {
        // parse the rules file if we were given one
        let rules = self
            .rules_file
            .as_ref()
            .map(|rules_file| parse_rules_file(rules_file, &self.format, &conf.keys.api))
            .transpose()?;
        // unwrap the outer option
        let (ingress, egress) = match rules {
            Some(rules) => (rules.ingress, rules.egress),
            None => (None, None),
        };
        // create the network policy request
        Ok(NetworkPolicyRequest {
            name: self.name.clone(),
            groups: (!self.groups.is_empty())
                .then_some(self.groups.clone())
                .ok_or(Error::new("Groups cannot be empty!"))?,
            ingress,
            egress,
            forced_policy: self.forced,
            default_policy: self.default,
        })
    }
}

impl UpdateNetworkPolicy {
    /// Generate a [`NetworkPolicyUpdate`] from the options set in a
    /// `UpdateNetworkPolicy` command
    fn to_update(&self, conf: &CtlConf) -> Result<NetworkPolicyUpdate, Error> {
        // parse the rules file
        let (add_ingress, add_egress) = match &self.opts.rules_file {
            Some(rules_file) => {
                let rules = parse_rules_file(rules_file, &self.opts.format, &conf.keys.api)?;
                (
                    rules.ingress.unwrap_or_default(),
                    rules.egress.unwrap_or_default(),
                )
            }
            None => (vec![], vec![]),
        };
        Ok(NetworkPolicyUpdate {
            new_name: self.opts.new_name.clone(),
            add_groups: self.opts.add_groups.clone(),
            remove_groups: self.opts.remove_groups.clone(),
            add_ingress,
            remove_ingress: self.opts.remove_ingress.clone(),
            clear_ingress: self.opts.clear_ingress,
            deny_all_ingress: self.opts.deny_all_ingress,
            add_egress,
            remove_egress: self.opts.remove_egress.clone(),
            deny_all_egress: self.opts.deny_all_egress,
            clear_egress: self.opts.clear_egress,
            forced_policy: self.opts.forced,
            default_policy: self.opts.default,
        })
    }
}

/// Verify a file containing rules or base network policies is valid
///
/// # Arguments
///
/// * `cmd` - The verify network policies command that was run
fn verify(cmd: &VerifyNetworkPolicies) -> Result<(), Error> {
    match cmd {
        VerifyNetworkPolicies::Rules(cmd) => {
            // open the rules file
            let file = std::fs::File::open(&cmd.rules_file).map_err(|err| {
                Error::new(format!(
                    "Unable to open rules file '{}': {}",
                    cmd.rules_file.to_string_lossy(),
                    err
                ))
            })?;
            // parse the rules file depending on the format
            let rules: NetworkPolicyRules = match cmd.format {
                NetworkPolicyFileFormat::Yaml => serde_yaml::from_reader(&file).map_err(|err| {
                    Error::new(format!(
                        "Invalid rules file '{}': {}",
                        cmd.rules_file.to_string_lossy(),
                        err,
                    ))
                })?,
                NetworkPolicyFileFormat::Json => serde_json::from_reader(&file).map_err(|err| {
                    Error::new(format!(
                        "Invalid rules file '{}': {}",
                        cmd.rules_file.to_string_lossy(),
                        err,
                    ))
                })?,
            };
            // print what we successfully parsed
            println!("Rules:\n\n{rules:#?}");
        }
        VerifyNetworkPolicies::Base(cmd) => {
            // open the base network policies file
            let file = std::fs::File::open(&cmd.policies_file).map_err(|err| {
                Error::new(format!(
                    "Unable to open base policies file '{}': {}",
                    cmd.policies_file.to_string_lossy(),
                    err
                ))
            })?;
            // parse the base network policies from YAML
            let base_policies: Vec<BaseNetworkPolicy> =
                serde_yaml::from_reader(&file).map_err(|err| {
                    Error::new(format!(
                        "Invalid base network policies file '{}': {}",
                        cmd.policies_file.to_string_lossy(),
                        err,
                    ))
                })?;
            // print what we successfully parsed
            println!("Base Network Policies:\n\n{base_policies:#?}");
        }
    }
    Ok(())
}

pub async fn handle(args: &Args, network_policies: &NetworkPolicies) -> Result<(), Error> {
    // load our config and instance our client
    let (conf, thorium) = utils::get_client(args).await?;
    match network_policies {
        NetworkPolicies::Get(cmd) => get(thorium, cmd).await,
        NetworkPolicies::Default(cmd) => get_default(thorium, cmd).await,
        NetworkPolicies::Describe(cmd) => describe(thorium, cmd).await,
        NetworkPolicies::Create(cmd) => create(thorium, conf, cmd).await,
        NetworkPolicies::Delete(cmd) => delete(thorium, cmd).await,
        NetworkPolicies::Update(cmd) => update(thorium, conf, cmd).await,
        NetworkPolicies::Verify(cmd) => verify(cmd),
    }
}
