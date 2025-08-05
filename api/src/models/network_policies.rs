//! The structure for network policies

use chrono::{DateTime, Utc};
use cidr::{AnyIpCidr, Ipv4Cidr, Ipv6Cidr};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::str::FromStr;
use uuid::Uuid;

use crate::Error;

/// A specific protocol type to allow on a port
#[derive(
    Debug,
    Clone,
    Serialize,
    Deserialize,
    PartialEq,
    strum::Display,
    strum::EnumString,
    schemars::JsonSchema,
)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum NetworkProtocol {
    /// The TCP protocol
    TCP,
    /// The UDP protocol
    UDP,
    /// The SCTP protocol
    SCTP,
}

/// A policy detailing one or more ports to allow traffic on and which [`Protocol`]
/// to allow on those port(s)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NetworkPolicyPort {
    /// The port to allow, or the first port in a range of ports to
    /// allow when used in conjunction with [`PortPolicy::end_port`]
    pub port: u16,
    /// An end port if specifying a range of ports; the end port is the
    /// last port in the range
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_port: Option<u16>,
    /// The protocol to allow on the specified port(s)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol: Option<NetworkProtocol>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Ipv4Block {
    /// The IP CIDR to allow
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub cidr: Ipv4Cidr,
    /// A subset of the [`Ipv4Block::cidr`] to exclude
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub except: Option<Vec<Ipv4Cidr>>,
}

impl Ipv4Block {
    /// Create a block of IPv4 addresses
    ///
    /// # Arguments
    ///
    /// * `cidr` - The IPv4 CIDR the block includes
    /// * `except` - An optional subset of the IPv4 CIDR to exclude from the block
    #[must_use]
    pub fn new(cidr: Ipv4Cidr, except: Option<Vec<Ipv4Cidr>>) -> Self {
        Self { cidr, except }
    }

    /// Create an IPv4 block with only a single address
    ///
    /// # Arguments
    ///
    /// * `ip` - The single IP address the block will contain
    #[must_use]
    pub fn new_ip(ip: Ipv4Addr) -> Self {
        Self {
            cidr: Ipv4Cidr::new_host(ip),
            except: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct Ipv6Block {
    /// The IP CIDR to allow
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub cidr: Ipv6Cidr,
    /// A subset of the [`Ipv6Block::cidr`] to exclude
    #[serde(default)]
    #[cfg_attr(feature = "api", schema(value_type = String))]
    pub except: Option<Vec<Ipv6Cidr>>,
}

impl Ipv6Block {
    /// Create a block of IPv6 addresses
    ///
    /// # Arguments
    ///
    /// * `cidr` - The IPv6 CIDR the block includes
    /// * `except` - An optional subset of the IPv6 CIDR to exclude from the block
    #[must_use]
    pub fn new(cidr: Ipv6Cidr, except: Option<Vec<Ipv6Cidr>>) -> Self {
        Self { cidr, except }
    }

    /// Create an IPv6 block with only a single address
    ///
    /// # Arguments
    ///
    /// * `ip` - The single IP address the block will contain
    #[must_use]
    pub fn new_ip(ip: Ipv6Addr) -> Self {
        Self {
            cidr: Ipv6Cidr::new_host(ip),
            except: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub enum IpBlock {
    V4(Ipv4Block),
    V6(Ipv6Block),
}

/// A custom label to use for matching network policies to [`NetworkPolicyCustomK8sRule`]s
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NetworkPolicyCustomLabel {
    /// The label's key
    pub key: String,
    /// The label's value
    pub value: String,
}

impl NetworkPolicyCustomLabel {
    /// Create a new custom label
    ///
    /// # Arguments
    ///
    /// * `key` - The label's key
    /// * `value` - The label's value
    pub fn new<T, S>(key: T, value: S) -> Self
    where
        T: Into<String>,
        S: Into<String>,
    {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

/// A custom rule to allow network access to/from namespaces/pods in K8's by namespace/pod label(s)
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, schemars::JsonSchema)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NetworkPolicyCustomK8sRule {
    /// A list of custom labels to match namespaces on
    ///
    /// These labels are matched by *AND* logic, meaning
    /// namespaces must have all of the given labels to match
    #[serde(default)]
    pub namespace_labels: Option<Vec<NetworkPolicyCustomLabel>>,
    /// A list of custom labels to match pods on
    ///
    /// These labels are matched by *AND* logic, meaning
    /// pods must have all of the given labels to match
    #[serde(default)]
    pub pod_labels: Option<Vec<NetworkPolicyCustomLabel>>,
}

impl NetworkPolicyCustomK8sRule {
    /// Create a new custom K8's rule
    ///
    /// # Arguments
    ///
    /// * `namespace_labels` - The labels to use to match namespaces the policy should apply to
    /// * `pod_labels` - The labels to use to match pods the policy should apply to
    #[must_use]
    pub fn new(
        namespace_labels: Option<Vec<NetworkPolicyCustomLabel>>,
        pod_labels: Option<Vec<NetworkPolicyCustomLabel>>,
    ) -> Self {
        Self {
            namespace_labels,
            pod_labels,
        }
    }
}

/// Specific network policy settings to apply to a tool, whether for
/// ingress or egress
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NetworkPolicyRule {
    /// This rule's ID
    pub id: Uuid,
    /// The list of IP's to allow access to or from
    ///
    /// Restrictions on ports from [`NetworkPolicySettings::ports`] still apply
    #[serde(default)]
    pub allowed_ips: Vec<IpBlock>,
    /// The list of Thorium groups (namespaces) to allow access to or from
    ///
    /// Restrictions on ports from [`NetworkPolicySettings::ports`] still apply
    #[serde(default)]
    pub allowed_groups: Vec<String>,
    /// The list of Thorium tools within the restricted tool's own group to allow access to or from
    ///
    /// Restrictions on ports from [`NetworkPolicySettings::ports`] still apply
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    /// Whether or not to allow blanket access for local IP addresses, allowing
    /// traffic to or from all private IP addresses
    ///
    /// Restrictions on ports from [`NetworkPolicySettings::ports`] still apply
    #[serde(default)]
    pub allowed_local: bool,
    /// Whether or not to allow blanket internet access, allowing traffic to or from
    /// all public IP addresses. This also allows access to the K8's `CoreDNS` service
    /// to allow for resolving domain names
    ///
    /// Restrictions on ports from [`NetworkPolicySettings::ports`] still apply
    #[serde(default)]
    pub allowed_internet: bool,
    /// Allow traffic to or from any entity
    ///
    /// Overrides all other settings except [`NetworkPolicySettings::ports`]
    #[serde(default)]
    pub allowed_all: bool,
    /// A list of specific ports + protocols to allow
    ///
    /// If empty, traffic will be allowed on all ports + protocols
    #[serde(default)]
    pub ports: Vec<NetworkPolicyPort>,
    /// A list of custom rules allowing access to peers on K8's,
    /// matched by namespace and/or pod label(s)
    #[serde(default)]
    pub allowed_custom: Vec<NetworkPolicyCustomK8sRule>,
}

impl NetworkPolicyRule {
    /// Add a CIDR as a raw string to allow communication to or from
    /// and optionally one or more subsets of that CIDR to exclude
    ///
    /// # Errors
    ///
    /// Returns an error if any of the given CIDR's are invalid and cannot be parsed
    /// *or* if the CIDR's aren't all of the same IP version
    ///
    /// # Arguments
    ///
    /// * `cidr` - The raw CIDR to parse
    /// * `except_cidrs` - Any subsets of that CIDR to exclude
    pub fn raw_cidr<T: AsRef<str>>(
        mut self,
        cidr: T,
        except_cidrs: Option<Vec<T>>,
    ) -> Result<Self, Error> {
        // parse the cidr
        let any_cidr = AnyIpCidr::from_str(cidr.as_ref()).map_err(|err| {
            Error::new(format!("Unable to parse CIDR '{}': {}", cidr.as_ref(), err))
        })?;
        // parse all the except cidr's
        let any_excepts = except_cidrs
            .map(|except| {
                except
                    .iter()
                    .map(|raw_cidr| AnyIpCidr::from_str(raw_cidr.as_ref()))
                    .collect::<Result<Vec<AnyIpCidr>, _>>()
            })
            .transpose()
            .map_err(|err| Error::new(format!("Error parsing except CIDR's: {err}")))?;
        // check if we're IPv4 or IPv6
        match any_cidr {
            AnyIpCidr::Any => {
                return Err(Error::new(format!(
                    "Unable to parse CIDR '{}': CIDR is detected as neither IPv4 or IPv6",
                    cidr.as_ref()
                )))
            }
            AnyIpCidr::V4(cidr_v4) => {
                let except_v4 = any_excepts
                    .map(|any_excepts| {
                        any_excepts
                            .into_iter()
                            .map(|any_except| match any_except {
                                AnyIpCidr::Any => Err(Error::new(
                                    "one or more except CIDR's is neither Ipv4 nor IPv6",
                                )),
                                AnyIpCidr::V4(except_v4) => Ok(except_v4),
                                AnyIpCidr::V6(_) => Err(Error::new(
                                    "CIDR is IPv4 but one or more except CIDR's is IPv6",
                                )),
                            })
                            .collect::<Result<Vec<Ipv4Cidr>, Error>>()
                    })
                    .transpose()
                    .map_err(|err| {
                        Error::new(format!(
                            "Error parsing except CIDR's: {}",
                            err.msg().unwrap_or_default()
                        ))
                    })?;
                self.allowed_ips.push(IpBlock::V4(Ipv4Block {
                    cidr: cidr_v4,
                    except: except_v4,
                }));
            }
            AnyIpCidr::V6(cidr_v6) => {
                let except_v6 = any_excepts
                    .map(|any_excepts| {
                        any_excepts
                            .into_iter()
                            .map(|any_except| match any_except {
                                AnyIpCidr::Any => Err(Error::new(
                                    "one or more except CIDR's is neither Ipv4 nor IPv6",
                                )),
                                AnyIpCidr::V4(_) => Err(Error::new(
                                    "CIDR is IPv6 but one or more except CIDR's is IPv4",
                                )),
                                AnyIpCidr::V6(except_v6) => Ok(except_v6),
                            })
                            .collect::<Result<Vec<Ipv6Cidr>, Error>>()
                    })
                    .transpose()
                    .map_err(|err| {
                        Error::new(format!(
                            "Error parsing except CIDR's: {}",
                            err.msg().unwrap_or_default()
                        ))
                    })?;
                self.allowed_ips.push(IpBlock::V6(Ipv6Block {
                    cidr: cidr_v6,
                    except: except_v6,
                }));
            }
        }
        Ok(self)
    }
}

/// Similar to [`IpBlock`] except it uses `String`s to
/// represent CIDR's rather than actual CIDR structs
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq, schemars::JsonSchema)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct IpBlockRaw {
    /// The IP CIDR to allow
    pub cidr: String,
    /// A subset of the [`IpBlockConf::cidr`] to exclude
    #[serde(default)]
    pub except: Option<Vec<String>>,
}

/// Specific network policy settings to apply to a tool, whether for
/// ingress or egress
///
/// Similar to [`NetworkPolicyRule`] except that it uses raw `String`s
/// instead of real CIDR's from the [`cidr`] crate for easier serialization and in order
/// to implement [`JsonSchema`] required for all fields in [`thorium::Conf`];
/// raw CIDR's are parsed when converted to a full-fledged [`NetworkPolicyRule`]
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq, schemars::JsonSchema)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NetworkPolicyRuleRaw {
    /// The list of IP's to allow access to or from
    ///
    /// Restrictions on ports from [`NetworkPolicySettings::ports`] still apply
    #[serde(default)]
    pub allowed_ips: Vec<IpBlockRaw>,
    /// The list of Thorium groups (namespaces) to allow access to or from
    ///
    /// Restrictions on ports from [`NetworkPolicySettings::ports`] still apply
    #[serde(default)]
    pub allowed_groups: Vec<String>,
    /// The list of Thorium tools within the restricted tool's own group to allow access to or from
    ///
    /// Restrictions on ports from [`NetworkPolicySettings::ports`] still apply
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    /// Whether or not to allow blanket access for local IP addresses, allowing
    /// traffic to or from all private IP addresses
    ///
    /// Restrictions on ports from [`NetworkPolicySettings::ports`] still apply
    #[serde(default)]
    pub allowed_local: bool,
    /// Whether or not to allow blanket internet access, allowing traffic to or from
    /// all public IP addresses. This also allows access to the K8's `CoreDNS` service
    /// to allow for resolving domain names
    ///
    /// Restrictions on ports from [`NetworkPolicySettings::ports`] still apply
    #[serde(default)]
    pub allowed_internet: bool,
    /// Allow traffic to or from any entity
    ///
    /// Overrides all other settings except [`NetworkPolicySettings::ports`]
    #[serde(default)]
    pub allowed_all: bool,
    /// A list of specific ports + protocols to allow
    ///
    /// If empty, traffic will be allowed on all ports + protocols
    #[serde(default)]
    pub ports: Vec<NetworkPolicyPort>,
    /// A list of custom rules allowing access to peers on K8's,
    /// matched by namespace and/or pod label(s)
    #[serde(default)]
    pub allowed_custom: Vec<NetworkPolicyCustomK8sRule>,
}

impl NetworkPolicyRuleRaw {
    /// Add a single IP address that can communicate
    ///
    /// # Arguments
    ///
    /// * `ip` - The IP address that can communicate
    #[must_use]
    pub fn ip(mut self, ip: IpAddr) -> Self {
        // add the ip as a block with just the address
        self.allowed_ips.push(IpBlockRaw {
            cidr: ip.to_string(),
            except: None,
        });
        self
    }

    /// Add an IP block as a raw string CIDR to allow communication to or from
    /// and optionally one or more subsets of that CIDR to exclude
    ///
    /// # Arguments
    ///
    /// * `cidr` - The raw CIDR to parse
    /// * `except_cidrs` - Any subsets of that CIDR to exclude
    #[must_use]
    pub fn ip_block<T: Into<String>>(mut self, cidr: T, except_cidrs: Option<Vec<T>>) -> Self {
        // add the ip block
        self.allowed_ips.push(IpBlockRaw {
            cidr: cidr.into(),
            except: except_cidrs.map(|except| except.into_iter().map(Into::into).collect()),
        });
        self
    }

    /// Add a group (namespace) that can communicate for this rule
    ///
    /// # Arguments
    ///
    /// * `group` - The group to add
    #[must_use]
    pub fn group<T: Into<String>>(mut self, group: T) -> Self {
        self.allowed_groups.push(group.into());
        self
    }

    /// Add groups (namespaces) that can communicate for this rule
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to add
    #[must_use]
    pub fn groups<I, T>(mut self, groups: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.allowed_groups
            .extend(groups.into_iter().map(Into::into));
        self
    }

    /// Add a tool that this tool can communicate with within its own group for this rule
    ///
    /// # Arguments
    ///
    /// * `tool` - The tool to add
    #[must_use]
    pub fn tool<T: Into<String>>(mut self, tool: T) -> Self {
        self.allowed_tools.push(tool.into());
        self
    }

    /// Add tools that this tool can communicate with within its own group for this rule
    ///
    /// # Arguments
    ///
    /// * `tools` - The tools to add
    #[must_use]
    pub fn tools<I, T>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.allowed_tools.extend(tools.into_iter().map(Into::into));
        self
    }

    /// Allow communication with all addresses in the private IP
    /// address space for this rule
    #[must_use]
    pub fn allow_local(mut self) -> Self {
        self.allowed_local = true;
        self
    }

    /// Allow communication with all addresses in the public IP
    /// address space for this rule
    #[must_use]
    pub fn allow_internet(mut self) -> Self {
        self.allowed_internet = true;
        self
    }

    /// Allow communication with all entities for this rule
    ///
    /// This setting overrides all other settings except ports
    #[must_use]
    pub fn allow_all(mut self) -> Self {
        self.allowed_all = true;
        self
    }

    /// Add a port or range of ports that this rule applies to
    ///
    /// # Arguments
    ///
    /// * `port` - The port (or first port if a range) that allows communication
    /// * `end_port` - The last port in the range that of ports that allows communication;
    ///                when specified, `port` is the first port in the range and `end_port` is the last
    /// * `protocol` - The protocol allowed on the port(s); if `None`, all protocols are allowed
    #[must_use]
    pub fn port(
        mut self,
        port: u16,
        end_port: Option<u16>,
        protocol: Option<NetworkProtocol>,
    ) -> Self {
        let port_policy = NetworkPolicyPort {
            port,
            end_port,
            protocol,
        };
        self.ports.push(port_policy);
        self
    }

    /// Add a custom rule to allow access to peers in K8s using namespace and/or pod label(s)
    ///
    /// # Arguments
    ///
    /// * `custom_rule` - The custom rule to add
    #[must_use]
    pub fn custom_rule(mut self, custom_rule: NetworkPolicyCustomK8sRule) -> Self {
        self.allowed_custom.push(custom_rule);
        self
    }

    /// Add a custom rule to allow access to peers in K8s using namespace and/or pod label(s)
    ///
    /// # Arguments
    ///
    /// * `custom_rules` - The custom rules to add
    #[must_use]
    pub fn custom_rules<I>(mut self, custom_rules: I) -> Self
    where
        I: IntoIterator<Item = NetworkPolicyCustomK8sRule>,
    {
        self.allowed_custom.extend(custom_rules);
        self
    }
}

impl TryFrom<NetworkPolicyRuleRaw> for NetworkPolicyRule {
    type Error = crate::Error;

    /// Try to convert a raw policy rule from [`crate::Conf`] to a real `NetworkPolicyRule`;
    ///
    /// Fallible because the raw rule uses a raw String to represent CIDR's, and errors
    /// can occur when casting
    ///
    /// # Arguments
    ///
    /// * `raw_rule` - The raw network policy rule to try to convert
    fn try_from(raw_rule: NetworkPolicyRuleRaw) -> Result<Self, Self::Error> {
        // create a blank policy rule to add allowed ips to
        let mut policy_rule = Self::default();
        for raw_ip_block in raw_rule.allowed_ips {
            // cast the raw cidrs in the raw ip block to real cidrs by adding them to
            // the blank policy
            policy_rule = policy_rule.raw_cidr(raw_ip_block.cidr, raw_ip_block.except)?;
        }
        // add the casted cidrs and the rest of the fields to a new rule
        Ok(Self {
            id: Uuid::new_v4(),
            allowed_ips: policy_rule.allowed_ips,
            allowed_groups: raw_rule.allowed_groups,
            allowed_tools: raw_rule.allowed_tools,
            allowed_local: raw_rule.allowed_local,
            allowed_internet: raw_rule.allowed_internet,
            allowed_all: raw_rule.allowed_all,
            ports: raw_rule.ports,
            allowed_custom: raw_rule.allowed_custom,
        })
    }
}

/// A request to create a [`NetworkPolicy`]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NetworkPolicyRequest {
    /// The name of the network policy
    pub name: String,
    /// The groups the policy is in
    pub groups: Vec<String>,
    /// The rules for ingress to the tool
    ///
    /// If None, all ingress traffic is allowed; if empty, no ingress traffic is allowed
    pub ingress: Option<Vec<NetworkPolicyRuleRaw>>,
    /// The rules for egress from the tool
    ///
    /// If None, all egress traffic is allowed; if empty, no egress traffic is allowed
    pub egress: Option<Vec<NetworkPolicyRuleRaw>>,
    /// This network policy should always be applied to all tools spawned in its group(s)
    ///
    /// Forced policies are not actually saved to individual images and are applied in the
    /// scaler when images are spawned
    #[serde(default)]
    pub forced_policy: bool,
    /// This policy should be added by default if none are given when creating an image
    /// in the policy's group(s)
    ///
    /// Default policies are saved to all new images in their groups and can be seen in
    /// in the images' data just like any other policy. Unlike forced policies, default
    /// policies are "applied" in the API on image creation, so editing a policy to no
    /// longer be default in its groups will not remove the policy from images
    #[serde(default)]
    pub default_policy: bool,
}

impl NetworkPolicyRequest {
    /// Create a request to create a new [`NetworkPolicy`]
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the network policy
    /// * `groups` - The groups the network policy is in
    pub fn new<S, I, T>(name: S, groups: I) -> Self
    where
        S: Into<String>,
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        Self {
            name: name.into(),
            groups: groups.into_iter().map(Into::into).collect(),
            ingress: None,
            egress: None,
            forced_policy: false,
            default_policy: false,
        }
    }

    /// Add an ingress rule to this network policy
    ///
    /// # Arguments
    ///
    /// * `rule` - The ingress rule to add
    #[must_use]
    pub fn add_ingress_rule(mut self, rule: NetworkPolicyRuleRaw) -> Self {
        self.ingress.get_or_insert(Vec::new()).push(rule);
        self
    }

    /// Denies all ingress traffic by setting `ingress` to an empty Vec
    #[must_use]
    pub fn deny_all_ingress(mut self) -> Self {
        self.ingress = Some(Vec::new());
        self
    }

    /// Add an egress rule to this network policy
    ///
    /// # Arguments
    ///
    /// * `rule` - The egress rule to add
    #[must_use]
    pub fn add_egress_rule(mut self, rule: NetworkPolicyRuleRaw) -> Self {
        self.egress.get_or_insert(Vec::new()).push(rule);
        self
    }

    /// Denies all egress traffic by setting `egress` to an empty Vec
    #[must_use]
    pub fn deny_all_egress(mut self) -> Self {
        self.egress = Some(Vec::new());
        self
    }

    /// This network policy should always be applied to all tools spawned in its group(s)
    #[must_use]
    pub fn forced_policy(mut self) -> Self {
        self.forced_policy = true;
        self
    }

    /// This policy should be added by default if others are not given for all tools in its group(s)
    #[must_use]
    pub fn default_policy(mut self) -> Self {
        self.default_policy = true;
        self
    }
}

/// An update to apply to a network policy
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NetworkPolicyUpdate {
    /// A new name for the network policy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_name: Option<String>,
    /// The groups to add the policy to
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub add_groups: Vec<String>,
    /// The groups to remove the policy from
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remove_groups: Vec<String>,
    /// The ingress rules to add
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub add_ingress: Vec<NetworkPolicyRuleRaw>,
    /// The ingress rules to remove
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remove_ingress: Vec<Uuid>,
    /// Clear all restrictions on ingress traffic from this network policy,
    /// clearing all ingress rules and setting ingress to None
    ///
    /// Overrides any settings to add/remove ingress rules;
    /// incompatible with `deny_all_ingress`
    #[serde(default)]
    pub clear_ingress: bool,
    /// Allow all ingress traffic by removing all ingress rules and
    /// setting ingress to None
    ///
    /// Overrides any settings to add/remove ingress rules;
    /// incompatible with `clear_ingress`
    #[serde(default)]
    pub deny_all_ingress: bool,
    /// The egress rules to add
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub add_egress: Vec<NetworkPolicyRuleRaw>,
    /// The egress rules to remove
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remove_egress: Vec<Uuid>,
    /// Clear all restrictions on egress traffic from this network policy,
    /// clearing all egress rules and setting egress to None
    ///
    /// Overrides any settings to add/remove egress rules;
    /// incompatible with `deny_all_egress`
    #[serde(default)]
    pub clear_egress: bool,
    /// Allow all egress traffic by removing all egress rules and
    /// setting egress to None
    ///
    /// Overrides any settings to add/remove egress rules;
    /// incompatible with `clear_egress`
    #[serde(default)]
    pub deny_all_egress: bool,
    /// Set whether or not this policy should always be applied to all tools spawned in its group(s)
    ///
    /// Forced policies are not actually saved to individual images and are simply applied
    /// in the scaler when images are spawned
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forced_policy: Option<bool>,
    /// Set whether or not this policy should be added by default if none are given when
    /// creating an image in the policy's group(s)
    ///
    /// Default policies are saved to all new images in their groups and can be seen in
    /// in the images' data just like any other policy. Unlike forced policies, default
    /// policies are "applied" in the API on image creation, so editing a policy to no
    /// longer be default in its groups will not automatically remove the policy from
    /// images it originally applied to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_policy: Option<bool>,
}

impl NetworkPolicyUpdate {
    /// Check to see if the update is empty
    #[allow(dead_code)]
    pub(crate) fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Set a new name for the network policy
    ///
    /// This name will be automatically updated for all images using the policy
    ///
    /// # Arguments
    ///
    /// * `name` - The new name for the network policy
    #[must_use]
    pub fn new_name<T: Into<String>>(mut self, new_name: T) -> Self {
        self.new_name = Some(new_name.into());
        self
    }

    /// Add a group to add the policy to
    ///
    /// # Arguments
    ///
    /// * `group` - The group the policy should be added to
    #[must_use]
    pub fn add_group<T: Into<String>>(mut self, group: T) -> Self {
        self.add_groups.push(group.into());
        self
    }

    /// Add groups to add the policy to
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups the policy should be added to
    #[must_use]
    pub fn add_groups<T, I>(mut self, groups: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        self.add_groups.extend(groups.into_iter().map(Into::into));
        self
    }

    /// Add a group to remove the policy from
    ///
    /// # Arguments
    ///
    /// * `group` - The group the policy should be removed from
    #[must_use]
    pub fn remove_group<T: Into<String>>(mut self, group: T) -> Self {
        self.remove_groups.push(group.into());
        self
    }

    /// Add groups to remove the policy from
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups the policy should be removed from
    #[must_use]
    pub fn remove_groups<T, I>(mut self, groups: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = T>,
    {
        self.remove_groups
            .extend(groups.into_iter().map(Into::into));
        self
    }

    /// Add an ingress rule to add to the policy
    ///
    /// # Arguments
    ///
    /// * `rule` - The rule to add to the policy
    #[must_use]
    pub fn add_ingress_rule(mut self, rule: NetworkPolicyRuleRaw) -> Self {
        self.add_ingress.push(rule);
        self
    }

    /// Add ingress rules to add to the policy
    ///
    /// # Arguments
    ///
    /// * `rules` - The rules to add to the policy
    #[must_use]
    pub fn add_ingress_rules<I>(mut self, rules: I) -> Self
    where
        I: IntoIterator<Item = NetworkPolicyRuleRaw>,
    {
        self.add_ingress.extend(rules);
        self
    }

    /// Add an ingress rule to remove from the policy
    ///
    /// # Arguments
    ///
    /// * `rule_id` - The ID of the rule to remove from the policy
    #[must_use]
    pub fn remove_ingress_rule(mut self, rule_id: Uuid) -> Self {
        self.remove_ingress.push(rule_id);
        self
    }

    /// Add ingress rules to remove from the policy
    ///
    /// # Arguments
    ///
    /// * `rule_ids` - The ID's of rules to remove from the policy
    #[must_use]
    pub fn remove_ingress_rules<I>(mut self, rule_ids: I) -> Self
    where
        I: IntoIterator<Item = Uuid>,
    {
        self.remove_ingress.extend(rule_ids);
        self
    }

    /// Remove any restrictions this network policy has on ingress by
    /// clearing all ingress rules and setting ingress to None
    ///
    /// Overrides any ingress rules to be added/removed; incompatible
    /// with `deny_all_ingress`
    #[must_use]
    pub fn clear_ingress(mut self) -> Self {
        self.clear_ingress = true;
        self
    }

    /// Deny all ingress traffic by clearing all ingress rules and setting
    /// ingress to an empty list
    ///
    /// Overrides any ingress rules to be added/removed; incompatible
    /// with `clear_ingress`
    #[must_use]
    pub fn deny_all_ingress(mut self) -> Self {
        self.deny_all_ingress = true;
        self
    }

    /// Add an egress rule to add to the policy
    ///
    /// # Arguments
    ///
    /// * `rule` - The rule to add to the policy
    #[must_use]
    pub fn add_egress_rule(mut self, rule: NetworkPolicyRuleRaw) -> Self {
        self.add_egress.push(rule);
        self
    }

    /// Add egress rules to add to the policy
    ///
    /// # Arguments
    ///
    /// * `rules` - The rules to add to the policy
    #[must_use]
    pub fn add_egress_rules<I>(mut self, rules: I) -> Self
    where
        I: IntoIterator<Item = NetworkPolicyRuleRaw>,
    {
        self.add_egress.extend(rules);
        self
    }

    /// Add an egress rule to remove from the policy
    ///
    /// # Arguments
    ///
    /// * `rule_id` - The ID of the rule to remove from the policy
    #[must_use]
    pub fn remove_egress_rule(mut self, rule_id: Uuid) -> Self {
        self.remove_egress.push(rule_id);
        self
    }

    /// Add egress rules to remove from the policy
    ///
    /// # Arguments
    ///
    /// * `rule_ids` - The ID's of rules to remove from the policy
    #[must_use]
    pub fn remove_egress_rules<I>(mut self, rule_ids: I) -> Self
    where
        I: IntoIterator<Item = Uuid>,
    {
        self.remove_egress.extend(rule_ids);
        self
    }

    /// Remove any restrictions this network policy has on egress by
    /// clearing all egress rules and setting egress to None
    ///
    /// Overrides any egress rules to be added/removed; incompatible
    /// with `deny_all_egress`
    #[must_use]
    pub fn clear_egress(mut self) -> Self {
        self.clear_egress = true;
        self
    }

    /// Deny all egress traffic by clearing all egress rules and setting
    /// egress to an empty list
    ///
    /// Overrides any egress rules to be added/removed; incompatible
    /// with `clear_egress`
    #[must_use]
    pub fn deny_all_egress(mut self) -> Self {
        self.deny_all_egress = true;
        self
    }

    /// Update whether or not the policy is a forced policy in its groups
    ///
    /// # Arguments
    ///
    /// * `forced_policy` - The forced policy setting to update to
    #[must_use]
    pub fn forced_policy(mut self, forced_policy: bool) -> Self {
        self.forced_policy = Some(forced_policy);
        self
    }

    /// Update whether or not the policy is a default policy in its groups
    ///
    /// # Arguments
    ///
    /// * `default_policy` - The default policy setting to update to
    #[must_use]
    pub fn default_policy(mut self, default_policy: bool) -> Self {
        self.default_policy = Some(default_policy);
        self
    }
}

/// The options that you can set when listing network policies in Thorium
#[derive(Debug, Clone)]
pub struct NetworkPolicyListOpts {
    /// The cursor to use to continue this search
    pub cursor: Option<Uuid>,
    /// The max number of objects to retrieve on a single page
    pub page_size: usize,
    /// The total number of objects to return with this cursor;
    /// if not set, data will be retrieved until the cursor is exhausted
    pub limit: Option<usize>,
    /// The groups to limit our search to
    pub groups: Vec<String>,
}

impl Default for NetworkPolicyListOpts {
    /// Build a default search
    fn default() -> Self {
        NetworkPolicyListOpts {
            cursor: None,
            page_size: 50,
            limit: None,
            groups: Vec::default(),
        }
    }
}

impl NetworkPolicyListOpts {
    /// Set the cursor to use when continuing this search
    ///
    /// # Arguments
    ///
    /// * `cursor` - The cursor id to use for this search
    #[must_use]
    pub fn cursor(mut self, cursor: Uuid) -> Self {
        // set cursor for this search
        self.cursor = Some(cursor);
        self
    }

    /// The max number of objects to retrieve in a single page
    ///
    /// # Arguments
    ///
    /// * `page_size` - The max number of documents to return in a single request
    #[must_use]
    pub fn page_size(mut self, page_size: usize) -> Self {
        // set the date to end listing files at
        self.page_size = page_size;
        self
    }

    /// Limit how many network policies this search can return in total
    ///
    /// # Arguments
    ///
    /// * `limit` - The max number of objects to return over the lifetime of this cursor
    #[must_use]
    pub fn limit(mut self, limit: usize) -> Self {
        // set the date to end listing files at
        self.limit = Some(limit);
        self
    }

    /// Add to a group to limit our search to
    ///
    /// # Arguments
    ///
    /// * `group` - A group to limit our search to
    #[must_use]
    pub fn group<T: Into<String>>(mut self, group: T) -> Self {
        self.groups.push(group.into());
        self
    }

    /// Add multiple groups to limit our search to
    ///
    /// # Arguments
    ///
    /// * `groups` - The groups to limit our search to
    #[must_use]
    pub fn groups<I, T>(mut self, groups: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.groups.extend(groups.into_iter().map(Into::into));
        self
    }
}

/// Set default for the network policy list limit
fn default_list_limit() -> usize {
    50
}

#[derive(Deserialize, Debug)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NetworkPolicyListParams {
    /// The cursor id to use if one exists
    pub cursor: Option<Uuid>,
    /// The max number of items to return in this response
    #[serde(default = "default_list_limit")]
    pub limit: usize,
    /// The groups to list data from
    #[serde(default)]
    pub groups: Vec<String>,
}

impl Default for NetworkPolicyListParams {
    /// Create a default network policy list params
    fn default() -> Self {
        Self {
            cursor: None,
            limit: default_list_limit(),
            groups: Vec::default(),
        }
    }
}

// A single sample submission line
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NetworkPolicyListLine {
    /// The group this network policy is apart of (used only for cursor generation)
    #[serde(skip_serializing, skip_deserializing)]
    pub groups: Vec<String>,
    /// The network policy's name
    pub name: String,
    /// The network policy's unique ID
    pub id: Uuid,
}

/// A Thorium Network Policy, currently mostly a wrapper for a Kubernetes
/// [Network Policy](https://kubernetes.io/docs/concepts/services-networking/network-policies)
///
/// A `NetworkPolicy` can selectively open communication between tools running in Thorium and other
/// entities in Thorium, K8's, the local network, or the internet. Network policies will only take
/// effect if the K8's cluster Thorium is running on has a supported network plugin installed.
///
/// Because network policies are additive, one or more restrictive base network policies applied
/// to all images are required for these policies to function properly. These base network policies
/// are set in the Thorium config file (see [`crate::Conf`]). If none are provided in the Thorium
/// config file, a default base policy will be applied that restricts all ingress traffic except
/// from the Thorium API and all egress traffic except to the Thorium API and on UDP port 53 to
/// the Kubernetes `CoreDNS` and `NodeLocalDNS` services as well as link-local address ranges
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "api", derive(utoipa::ToSchema))]
pub struct NetworkPolicy {
    /// The name of the network policy
    pub name: String,
    /// The network policy's unique ID
    pub id: Uuid,
    /// The network policy's name converted to something that K8's will accept,
    /// appended with its UUID (`<K8_NAME>-<UUID>`)
    pub k8s_name: String,
    /// The groups the network policy is available to
    pub groups: Vec<String>,
    /// The time this policy was created
    pub created: DateTime<Utc>,
    /// The rules for ingress to the tool
    ///
    /// If ingress is None, no rules are applied and all ingress traffic is allowed;
    /// if ingress is empty, all ingress traffic is denied
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ingress: Option<Vec<NetworkPolicyRule>>,
    /// The rules for egress from the tool
    ///
    /// If egress is None, no rules are applied and all egress traffic is allowed;
    /// if egress is empty, all egress traffic is denied
    #[serde(skip_serializing_if = "Option::is_none")]
    pub egress: Option<Vec<NetworkPolicyRule>>,
    /// This network policy should always be applied to all tools spawned in its group(s)
    ///
    /// Forced policies are not actually saved to individual images and are simply applied
    /// in the scaler when images are spawned
    pub forced_policy: bool,
    /// This policy should be added by default if none are given when creating an image
    /// in the policy's group(s)
    ///
    /// Default policies are saved to all new images in their groups and can be seen in
    /// in the images' data just like any other policy. Unlike forced policies, default
    /// policies are "applied" in the API on image creation, so editing a policy to no
    /// longer be default in its groups will not automatically remove the policy from
    /// images it originally applied to
    pub default_policy: bool,
    /// The list of images using this network policy mapped by group
    pub used_by: HashMap<String, Vec<String>>,
}

// add K8's api support for the scaler
cfg_if::cfg_if! {
    if #[cfg(feature = "k8s")] {
        use k8s_openapi::api::networking::v1 as k8s_net;
        use k8s_openapi::apimachinery::pkg::util::intstr::IntOrString;
        use k8s_openapi::apimachinery::pkg::apis::meta::v1::LabelSelector;
        use std::sync::LazyLock;

        impl NetworkPolicy {
            /// Returns true if there have been any changes to the network policy that
            /// require an update in K8's
            ///
            /// Notably, changes to the policy's name, creation date, used_by set, and
            /// forced/default status don't require a K8's update
            ///
            /// # Arguments
            ///
            /// * `old_policy` - The old version of the policy
            #[must_use]
            pub fn needs_k8s_update(&self, old_policy: &Self) -> bool {
                self.k8s_name != old_policy.k8s_name || self.groups != old_policy.groups || self.ingress != old_policy.ingress || self.egress != old_policy.egress
            }
        }

        /// A list of IP blocks representing all public IP addresses by specifying CIDR's matching
        /// all public addresses except CIDR lists that include all private IP addresses
        static PUBLIC_IP_K8S_BLOCKS: LazyLock<Vec<k8s_net::IPBlock>> = LazyLock::new(|| {
            vec! [
                // IPv4
                k8s_net::IPBlock {
                    cidr: "0.0.0.0/0".to_string(),
                    except: Some(vec![
                        "10.0.0.0/8".to_string(),
                        "172.16.0.0/12".to_string(),
                        "192.168.0.0/16".to_string(),
                    ])
                },
                // IPv6
                k8s_net::IPBlock {
                    cidr: "::/0".to_string(),
                    except: Some(vec![
                        "fc00::/7".to_string(),
                    ])
                }
            ]
        });
        /// A list of IP blocks representing local IP addresses
        static PRIVATE_IP_K8S_BLOCKS: LazyLock<Vec<k8s_net::IPBlock>> = LazyLock::new(|| {
            vec![
                // IPv4
                k8s_net::IPBlock {
                    cidr: "10.0.0.0/8".to_string(),
                    except: None,
                },
                k8s_net::IPBlock {
                    cidr: "172.16.0.0/12".to_string(),
                    except: None,
                },
                k8s_net::IPBlock {
                    cidr: "192.168.0.0/16".to_string(),
                    except: None,
                },
                // IPv6
                k8s_net::IPBlock {
                    cidr: "fc00::/7".to_string(),
                    except: None,
                },

            ]
        });

        impl From<IpBlock> for k8s_net::IPBlock {
            /// Convert a Thorium `IpBlock` to a [`k8s_net::IPBlock`]
            ///
            /// # Arguments
            ///
            /// * `thorium_ip_block` - The Thorium ip block to convert
            fn from(thorium_ip_block: IpBlock) -> Self {
                    match thorium_ip_block {
                        IpBlock::V4(ipv4_block) => {
                            Self {
                                cidr: ipv4_block.cidr.to_string(),
                                except: ipv4_block.except.map(|cidrs| cidrs.into_iter().map(|cidr| cidr.to_string()).collect()),
                            }
                        },
                        IpBlock::V6(ipv6_block) => {
                            Self {
                                cidr: ipv6_block.cidr.to_string(),
                                except: ipv6_block.except.map(|cidrs| cidrs.into_iter().map(|cidr| cidr.to_string()).collect()),
                            }
                        },
                    }
            }
        }

        impl From<NetworkPolicyPort> for k8s_net::NetworkPolicyPort {
            /// Convert a Thorium `NetworkPolicyPort` to a [`k8s_net::NetworkPolicyEgressRule`]
            ///
            /// # Arguments
            ///
            /// * `thorium_port` - The Thorium port to convert
            fn from(thorium_port: NetworkPolicyPort) -> Self {
                Self {
                    end_port: thorium_port.end_port.map(Into::into),
                    port: Some(IntOrString::Int(thorium_port.port.into())),
                    protocol: thorium_port.protocol.map(|p| p.to_string())
                }
            }
        }


        macro_rules! impl_from_rule {
            ($to_or_from:ident, $rule_type:ident) => {
                impl From<NetworkPolicyRule> for k8s_net::$rule_type {
                    /// Convert a Thorium `NetworkPolicyRule` to a [`k8s_net::NetworkPolicyEgressRule`]
                    ///
                    /// # Arguments
                    ///
                    /// * `thorium_rule` - The Thorium rule to convert
                    #[allow(clippy::too_many_lines)]
                    fn from(mut thorium_rule: NetworkPolicyRule) -> Self {
                        let peers: Vec<k8s_net::NetworkPolicyPeer> = if thorium_rule.allowed_all {
                            // provide an empty peers list to allow all peers
                            Vec::new()
                        } else {
                            // create a new peers list
                            let mut peers: Vec<k8s_net::NetworkPolicyPeer> = Vec::new();
                            if thorium_rule.allowed_local {
                                // add entire private ip address space if we allow local
                                peers.extend(
                                    PRIVATE_IP_K8S_BLOCKS
                                        .clone()
                                        .into_iter()
                                        .map(|ip_block| k8s_net::NetworkPolicyPeer {
                                            ip_block: Some(ip_block),
                                            namespace_selector: None,
                                            pod_selector: None,
                                        }
                                ));
                                // add kube node local DNS so we can get local domain names
                                peers.push(k8s_net::NetworkPolicyPeer {
                                    ip_block: None,
                                    namespace_selector: Some(LabelSelector {
                                        match_expressions: None,
                                        match_labels: Some(
                                            [("kubernetes.io/metadata.name".to_string(), "kube-system".to_string())]
                                                .into_iter()
                                                .collect()
                                            )
                                    }),
                                    pod_selector: Some(LabelSelector {
                                        match_expressions: None,
                                        match_labels: Some(
                                            [("k8s-app".to_string(), "node-local-dns".to_string())]
                                                .into_iter()
                                                .collect()
                                            )
                                    })
                                })
                            }
                            if thorium_rule.allowed_internet {
                                // add entire public ip address space if we allow internet
                                peers.extend(
                                    PUBLIC_IP_K8S_BLOCKS
                                        .clone()
                                        .into_iter()
                                        .map(|ip_block| k8s_net::NetworkPolicyPeer {
                                            ip_block: Some(ip_block),
                                            namespace_selector: None,
                                            pod_selector: None,
                                        }
                                ));
                                // add kube core DNS so we can get domain names
                                peers.push(k8s_net::NetworkPolicyPeer {
                                    ip_block: None,
                                    namespace_selector: Some(LabelSelector {
                                        match_expressions: None,
                                        match_labels: Some(
                                            [("kubernetes.io/metadata.name".to_string(), "kube-system".to_string())]
                                                .into_iter()
                                                .collect()
                                            )
                                    }),
                                    pod_selector: Some(LabelSelector {
                                        match_expressions: None,
                                        match_labels: Some(
                                            [("k8s-app".to_string(), "kube-dns".to_string())]
                                                .into_iter()
                                                .collect()
                                            )
                                    })
                                })
                            }
                            // add ips as ip blocks with no namespace/pod selectors
                            for ip_block in thorium_rule.allowed_ips.drain(..) {
                                peers.push(k8s_net::NetworkPolicyPeer {
                                    ip_block: Some(ip_block.into()),
                                    namespace_selector: None,
                                    pod_selector: None
                                });
                            }
                            // add groups as namespace label selectors
                            for group in thorium_rule.allowed_groups.drain(..) {
                                peers.push(
                                    k8s_net::NetworkPolicyPeer {
                                        ip_block: None,
                                        namespace_selector: Some(LabelSelector {
                                            match_expressions: None,
                                            // use the kubernetes automatic name label to target an entire namespace;
                                            // we could also match the "group" label on pods, but this seems simpler
                                            match_labels: Some(
                                                [("kubernetes.io/metadata.name".to_string(), group)]
                                                    .into_iter()
                                                    .collect()
                                                )
                                        }),
                                        pod_selector: None
                                    }
                                );
                            }
                            // add tools as pod label selectors
                            for tool in thorium_rule.allowed_tools.drain(..) {
                                peers.push(
                                    k8s_net::NetworkPolicyPeer {
                                        ip_block: None,
                                        namespace_selector: None,
                                        pod_selector: Some(LabelSelector {
                                            match_expressions: None,
                                            // use the "stage" label to select the tool
                                            match_labels: Some(
                                                [("stage".to_string(), tool)]
                                                    .into_iter()
                                                    .collect()
                                                )
                                        })
                                    }
                                );
                            }
                            // add any custom K8's rules
                            for mut custom_rule in thorium_rule.allowed_custom.drain(..) {
                                peers.push(
                                    k8s_net::NetworkPolicyPeer {
                                        ip_block: None,
                                        // add namespace labels if we have any
                                        namespace_selector: custom_rule.namespace_labels.as_mut().map(|labels| {
                                            (!labels.is_empty()).then_some(LabelSelector {
                                                match_expressions: None,
                                                match_labels: Some(labels
                                                    .drain(..)
                                                    .map(|label| (label.key, label.value))
                                                    .collect())
                                                })
                                            }).flatten(),
                                        // add pod labels if we have any
                                        pod_selector: custom_rule.pod_labels.as_mut().map(|labels| {
                                            (!labels.is_empty()).then_some(LabelSelector {
                                                match_expressions: None,
                                                match_labels: Some(labels
                                                    .drain(..)
                                                    .map(|label| (label.key, label.value))
                                                    .collect())
                                                })
                                            }).flatten(),
                                    }
                                );
                            }
                            peers
                        };
                        Self {
                            ports: (!thorium_rule.ports.is_empty()).then_some(
                                        thorium_rule.ports
                                            .into_iter()
                                            .map(Into::into)
                                            .collect()
                                    ),
                            $to_or_from: (!peers.is_empty()).then_some(peers),
                        }
                    }
                }
            };
        }

        // NetworkPolicyRule to NetworkPolicyIngressRule
        impl_from_rule!(from, NetworkPolicyIngressRule);
        // NetworkPolicyRule to NetworkPolicyEgressRule
        impl_from_rule!(to, NetworkPolicyEgressRule);
    }
}
