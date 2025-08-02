//! Arguments for network policy-related Thorctl commands

#![allow(clippy::module_name_repetitions)]

use serde::{Deserialize, Deserializer, Serialize};
use std::{path::PathBuf, str::FromStr};
use thorium::{models::NetworkPolicyRuleRaw, Error};

use clap::Parser;
use uuid::Uuid;

use super::{
    traits::{
        describe::DescribeSealed,
        search::{SearchParams, SearchSealed},
    },
    DescribeCommand, SearchParameterized,
};

/// The commands to send to the network policies task handler
#[derive(Debug, Parser)]
pub enum NetworkPolicies {
    /// List network policies in Thorium
    #[clap(version, author)]
    Get(GetNetworkPolicies),
    /// List the default network policies for a given group
    #[clap(version, author)]
    Default(DefaultNetworkPolicies),
    /// Describe one or more network policies, retrieving all details
    #[clap(version, author)]
    Describe(DescribeNetworkPolicies),
    /// Create a network policy in Thorium
    #[clap(version, author)]
    Create(CreateNetworkPolicy),
    /// Delete a network policy from Thorium
    #[clap(version, author)]
    Delete(DeleteNetworkPolicy),
    /// Update a network policy in Thorium
    #[clap(version, author)]
    Update(UpdateNetworkPolicy),
    /// Verify network policy rules files or base network policies for
    /// use in the Thorium cluster config
    #[clap(subcommand)]
    Verify(VerifyNetworkPolicies),
}

/// A command to list network policies
#[derive(Debug, Parser)]
pub struct GetNetworkPolicies {
    /// Any groups to filter by when listing network policies
    ///
    /// If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long)]
    pub groups: Vec<String>,
    /// The cursor to continue a search with
    #[clap(long)]
    pub cursor: Option<Uuid>,
    /// The max number of total network policies to find in the search
    #[clap(short, long, default_value_t = 50)]
    pub limit: usize,
    /// Refrain from setting a limit when listing network policies
    #[clap(long, conflicts_with = "limit")]
    pub no_limit: bool,
    /// The number of file to find in one request
    #[clap(short, long, default_value_t = 50)]
    pub page_size: usize,
}

impl SearchParameterized for GetNetworkPolicies {
    fn has_targets(&self) -> bool {
        false
    }

    fn apply_to_all(&self) -> bool {
        false
    }
}

impl SearchSealed for GetNetworkPolicies {
    fn get_search_params(&self) -> SearchParams {
        SearchParams {
            groups: &self.groups,
            tags: &[],
            tags_case_insensitive: false,
            delimiter: ' ',
            start: &None,
            end: &None,
            date_fmt: "",
            cursor: self.cursor,
            limit: self.limit,
            no_limit: self.no_limit,
            page_size: self.page_size,
        }
    }
}

/// A command to get the list of default network policies in a group
#[derive(Debug, Parser)]
pub struct DefaultNetworkPolicies {
    /// The group to get default network policies for
    pub group: String,
}

pub struct NetworkPolicyTarget {
    /// The network policy's name
    pub name: String,
    /// The network policy's unique ID in case more than one distinct network policies
    /// share the same name
    pub id: Option<Uuid>,
}

/// A command to describe one or more network policies
#[derive(Debug, Parser)]
pub struct DescribeNetworkPolicies {
    /// A list of network policies to describe optionally with an ID separated with a ':' in
    /// case more than one distinct network polices share the same name (i.e. <NAME:ID>)
    pub network_policies: Vec<String>,
    /// The delimiter character to use to separate network policy names and ID's
    #[clap(long, default_value_t = ':')]
    pub id_delimiter: char,
    /// The path to the file containing a list of network policies to describe separated by newlines
    /// optionally each with the policy's ID separated with a ':' in case more than one distinct network
    /// policies share the same name (i.e. <NAME:ID>)
    #[clap(long)]
    pub network_policy_list: Option<PathBuf>,
    /// The path to the file to write output to; if not provided, details will be output to stdout
    #[clap(short, long)]
    pub output: Option<PathBuf>,
    /// Output details in a condensed format (no formatting/whitespace)
    #[clap(long)]
    pub condensed: bool,
    /// Any groups to filter by when searching for network policies to describe
    ///
    /// If no groups are given, the search will include all groups the user is apart of
    #[clap(short, long, value_delimiter = ',')]
    pub groups: Vec<String>,
    /// The cursor to continue a search with
    #[clap(long)]
    pub cursor: Option<Uuid>,
    /// The max number of total network policies to find in the search
    #[clap(short, long, default_value_t = 50)]
    pub limit: usize,
    /// Refrain from setting a limit when listing network policies
    #[clap(long, conflicts_with = "limit")]
    pub no_limit: bool,
    /// The number of file to find in one request
    #[clap(short, long, default_value_t = 50)]
    pub page_size: usize,
    /// Refrain from setting any filters when describing network policies, attempting to describe all
    /// policies the user can view
    ///
    /// This will override any other search parameters set except for those associated with limit.
    /// When combined with `--no-limit`, this will describe all network policies to which the current user
    /// has access.
    #[clap(long)]
    pub describe_all: bool,
}

impl SearchParameterized for DescribeNetworkPolicies {
    fn has_targets(&self) -> bool {
        !self.network_policies.is_empty() || self.network_policy_list.is_some()
    }

    fn apply_to_all(&self) -> bool {
        self.describe_all
    }
}

impl SearchSealed for DescribeNetworkPolicies {
    fn get_search_params(&self) -> SearchParams {
        SearchParams {
            groups: &self.groups,
            tags: &[],
            tags_case_insensitive: false,
            delimiter: ' ',
            start: &None,
            end: &None,
            date_fmt: "",
            cursor: self.cursor,
            limit: self.limit,
            no_limit: self.no_limit,
            page_size: self.page_size,
        }
    }
}

impl DescribeSealed for DescribeNetworkPolicies {
    type Data = thorium::models::NetworkPolicy;

    type Target<'a> = NetworkPolicyTarget;

    type Cursor = thorium::models::Cursor<Self::Data>;

    fn raw_targets(&self) -> &[String] {
        &self.network_policies
    }

    fn condensed(&self) -> bool {
        self.condensed
    }

    fn out_path(&self) -> Option<&PathBuf> {
        self.output.as_ref()
    }

    fn target_list(&self) -> Option<&PathBuf> {
        self.network_policy_list.as_ref()
    }

    fn parse_target<'a>(&self, raw: &'a str) -> Result<Self::Target<'a>, thorium::Error> {
        // split on our delimiter
        let mut split = raw.split(self.id_delimiter);
        // try to parse the name
        let name = split.next().ok_or(
            Error::new("Invalid network policy target; network policy \
                must contain a name and optionally an ID delimited with the delimiter given in '--id-delimiter' \
                (e.g. <NAME>:<ID>)"))?;
        // see if we can parse an ID
        let id = match split.next() {
            Some(raw_id) => Some(Uuid::from_str(raw_id).map_err(|err| {
                Error::new(format!(
                    "Invalid network policy target: malformed ID - {err}"
                ))
            })?),
            None => None,
        };
        // error out if we have more data left
        if split.next().is_some() {
            return Err(Error::new("Invalid network policy target; network policy \
                must contain a name and optionally an ID delimited with the delimiter given in '--id-delimiter' \
                (e.g. <NAME>:<ID>)"));
        }
        Ok(NetworkPolicyTarget {
            name: name.to_owned(),
            id,
        })
    }

    async fn retrieve_data<'a>(
        &self,
        target: Self::Target<'a>,
        thorium: &thorium::Thorium,
    ) -> Result<Self::Data, thorium::Error> {
        thorium.network_policies.get(&target.name, target.id).await
    }

    async fn retrieve_data_search(
        &self,
        thorium: &thorium::Thorium,
    ) -> Result<Vec<Self::Cursor>, thorium::Error> {
        Ok(vec![
            thorium
                .network_policies
                .list_details(&self.build_network_policy_opts())
                .await?,
        ])
    }
}

impl DescribeCommand for DescribeNetworkPolicies {}

#[derive(Debug, Clone, Default, clap::ValueEnum)]
pub enum NetworkPolicyFileFormat {
    #[default]
    Yaml,
    Json,
}

#[derive(clap::Args, Debug, Clone)]
pub struct CreateNetworkPolicy {
    /// The name of the network policy
    #[clap(short, long)]
    pub name: String,
    /// The groups to add this network policy to
    #[clap(short, long, required = true, value_delimiter = ',')]
    pub groups: Vec<String>,
    /// The path to the JSON/YAML file defining the network policy's rules;
    /// if none provided, a network policy with no rules (allowing all traffic)
    /// will be created
    ///
    /// See the Thorium docs for guidance on rule-formatting
    #[clap(short = 'f', long)]
    pub rules_file: Option<PathBuf>,
    /// The format the network policy rules file is in
    #[clap(long, value_enum, default_value_t, ignore_case = true)]
    pub format: NetworkPolicyFileFormat,
    /// Sets the policy to be forcibly applied to all images in its group(s)
    #[clap(long, default_value_t)]
    pub forced: bool,
    /// Sets the policy to be a default policy for images in its group(s),
    /// added to new images when no other policies are given
    #[clap(long, default_value_t)]
    pub default: bool,
}

/// Deserialize to an empty Some value if none was explicitly provided
///
/// Allows us to deserialize to an empty Vec
fn deserialize_some<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    // map any value to Some if one was provided, even null
    Deserialize::deserialize(deserializer).map(Some)
}

/// The rules to add to a network policy
#[derive(Serialize, Deserialize, Debug)]
pub struct NetworkPolicyRules {
    /// The rules for ingress for the network policy
    #[serde(default, deserialize_with = "deserialize_some")]
    pub ingress: Option<Vec<NetworkPolicyRuleRaw>>,
    /// The rules for egress for the network policy
    #[serde(default, deserialize_with = "deserialize_some")]
    pub egress: Option<Vec<NetworkPolicyRuleRaw>>,
}

#[derive(clap::Args, Debug, Clone)]
pub struct DeleteNetworkPolicy {
    /// The name of the network policy to delete
    pub name: String,
    /// The network policy's ID, necessary if one or more distinct network policies
    /// share the same name
    #[clap(long)]
    pub id: Option<Uuid>,
}

#[derive(clap::Args, Debug, Clone)]
pub struct UpdateNetworkPolicy {
    /// The name of the network policy to update
    #[clap(long)]
    pub name: String,
    /// The network policy's ID, necessary if one or more distinct network policies
    /// share the same name
    #[clap(long)]
    pub id: Option<Uuid>,
    #[clap(flatten)]
    pub opts: NetworkPolicyUpdateOpts,
}

/// The set of possible updates to a Thorium `NetworkPolicy` where at least one is set
#[allow(clippy::struct_excessive_bools)]
#[derive(clap::Args, Debug, Clone)]
#[group(required = true, multiple = true)]
pub struct NetworkPolicyUpdateOpts {
    /// The new name to set for the network policy
    #[clap(long)]
    pub new_name: Option<String>,
    /// A list of groups to add the network policy to
    #[clap(long, value_delimiter = ',')]
    pub add_groups: Vec<String>,
    /// A list of groups to remove the network policy from
    #[clap(long, value_delimiter = ',')]
    pub remove_groups: Vec<String>,
    /// A list of ingress rules to remove from the policy
    ///
    /// Each network policy rule has a unique ID; the rule ID's can be found
    /// by describing the network policy
    #[clap(long, value_delimiter = ',')]
    pub remove_ingress: Vec<Uuid>,
    /// A list of egress rules to remove from the policy
    ///
    /// Each network policy rule has a unique ID; the rule ID's can be found
    /// by describing the network policy
    #[clap(long, value_delimiter = ',')]
    pub remove_egress: Vec<Uuid>,
    /// The path to the JSON/YAML file defining the network policy rules to add
    ///
    /// See the Thorium docs for guidance on rule-formatting
    #[clap(short = 'f', long)]
    pub rules_file: Option<PathBuf>,
    /// The format the network policy rules file is in
    #[clap(long, value_enum, default_value_t, ignore_case = true)]
    pub format: NetworkPolicyFileFormat,
    /// Clear all restrictions on ingress traffic from this network policy,
    /// clearing all ingress rules and setting ingress to None
    ///
    /// Overrides any settings to add/remove ingress rules;
    /// incompatible with `deny_all_ingress`
    #[clap(long, conflicts_with = "deny_all_ingress")]
    pub clear_ingress: bool,
    /// Deny all ingress and clear all ingress rules, setting ingress
    /// rules to an empty list
    ///
    /// Overrides any settings to add/remove ingress rules;
    /// incompatible with `clear_ingress`
    #[clap(long, conflicts_with = "clear_ingress")]
    pub deny_all_ingress: bool,
    /// Clear all restrictions on egress traffic from this network policy,
    /// clearing all egress rules and setting egress to None
    ///
    /// Overrides any settings to add/remove egress rules;
    /// incompatible with `deny_all_egress`
    #[clap(long, conflicts_with = "deny_all_egress")]
    pub clear_egress: bool,
    /// Deny all egress and clear all egress rules, setting egress
    /// rules to an empty list
    ///
    /// Overrides any settings to add/remove egress rules;
    /// incompatible with `clear_egress`
    #[clap(long, conflicts_with = "clear_egress")]
    pub deny_all_egress: bool,
    /// Set whether the policy should be forcibly applied to all images in its group(s)
    #[clap(long)]
    pub forced: Option<bool>,
    /// Set whether the policy is a default policy for images in its group(s),
    /// added to new images when no other policies are given
    #[clap(long)]
    pub default: Option<bool>,
}

/// Verify a network policy rules file or a base network policies file
#[derive(Debug, Parser)]
pub enum VerifyNetworkPolicies {
    /// Verify a network policy rules file
    Rules(VerifyNetworkPolicyRules),
    /// Verify a base network policies config
    Base(VerifyBaseNetworkPolicies),
}

/// The args for verifying a network policy rules file
#[derive(Debug, Parser)]
pub struct VerifyNetworkPolicyRules {
    /// The path to the network policy file to
    pub rules_file: PathBuf,
    /// The format the rules file is in
    #[clap(value_enum, short, long, default_value_t)]
    pub format: NetworkPolicyFileFormat,
}

/// The args for verifying a base network policies file
#[derive(Debug, Parser)]
pub struct VerifyBaseNetworkPolicies {
    /// The path to the file containing base network policies to verify
    pub policies_file: PathBuf,
}
