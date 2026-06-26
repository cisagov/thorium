//! Contains trait implementations necessary for testing but not used anywhere else

use std::collections::{BTreeSet, HashSet};
use uuid::Uuid;

use crate::models::helpers::matches_vecs_helper;
use crate::models::{
    CollectionEntity, CollectionEntityRequest, Country, DeviceEntity, DeviceEntityRequest, Entity,
    EntityMetadata, EntityMetadataRequest, EntityMetadataUpdate, EntityRequest, EntityUpdate, Group,
    GroupRequest, Image,
    ImageRequest, NetworkPolicy, NetworkPolicyRequest, NetworkPolicyRule, NetworkPolicyRuleRaw,
    NetworkPolicyUpdate, Pipeline, PipelineRequest, VendorEntity, VendorEntityRequest,
};
use crate::{
    matches_adds, matches_adds_iter, matches_clear, matches_clear_vec_opt, matches_removes,
    matches_removes_iter, matches_update, matches_update_opt, matches_vec, same,
};

impl PartialEq<Group> for GroupRequest {
    /// Check if a [`Group`] corresponds to a [`GroupRequest`]
    ///
    /// # Arguments
    ///
    /// * `group` - The `Group` to compare against
    fn eq(&self, group: &Group) -> bool {
        // make sure the name is the same
        same!(group.name, self.name);
        // make sure user types are the same
        same!(group.owners, self.owners);
        same!(group.managers, self.managers);
        same!(group.users, self.users);
        same!(group.monitors, self.monitors);
        same!(group.description, self.description);
        true
    }
}

impl PartialEq<Image> for ImageRequest {
    /// Check if an [`Image`] corresponds to an [`ImageRequest`]
    ///
    /// # Arguments
    ///
    /// * `image` - The `Image` to compare against
    fn eq(&self, image: &Image) -> bool {
        // make sure all fields are the same
        same!(image.name, self.name);
        same!(image.group, self.group);
        same!(&image.version, &self.version);
        same!(image.scaler, self.scaler);
        same!(image.image, self.image);
        same!(&image.lifetime, &self.lifetime);
        same!(image.timeout, self.timeout);
        same!(image.resources, self.resources);
        same!(image.spawn_limit, self.spawn_limit);
        same!(image.env, self.env);
        matches_vec!(&image.volumes, &self.volumes);
        same!(image.description, self.description);
        matches_update!(image.security_context, self.security_context);
        same!(image.collect_logs, self.collect_logs);
        same!(image.generator, self.generator);
        same!(image.dependencies, self.dependencies);
        same!(image.display_type, self.display_type);
        same!(image.output_collection, self.output_collection);
        same!(image.child_filters, self.child_filters);
        same!(image.network_policies, self.network_policies);
        true
    }
}

impl PartialEq<Pipeline> for PipelineRequest {
    /// Check if a [`Pipeline`] corresponds to a [`PipelineRequest`]
    ///
    /// # Arguments
    ///
    /// * `pipe` - The `Pipeline` to compare against
    fn eq(&self, pipe: &Pipeline) -> bool {
        // make sure all fields are the same
        same!(pipe.name, self.name);
        same!(pipe.group, self.group);
        same!(self.compare_order(&pipe.order), true);
        same!(&pipe.sla, self.sla.as_ref().unwrap_or(&604_800));
        same!(&pipe.triggers, &self.triggers);
        same!(&pipe.description, &self.description);
        true
    }
}

impl PartialEq<NetworkPolicyRuleRaw> for NetworkPolicyRule {
    /// Checks if all the info in a [`NetworkPolicyRuleRaw`] was set for a [`NetworkPolicyRule`]
    ///
    /// The only "gotcha" is that the conversion from `NetworkPolicyRuleRaw` to
    /// `NetworkPolicyRule` is fallible, so if a raw rule is invalid,
    /// the two are not equal by definition
    ///
    /// # Arguments
    ///
    /// * `raw_rule` - The raw rule to compare against
    fn eq(&self, raw_rule: &NetworkPolicyRuleRaw) -> bool {
        // try casting the raw rule
        let cast_result = NetworkPolicyRule::try_from(raw_rule.clone());
        match cast_result {
            Ok(mut cast) => {
                // set our id to the rule's id
                cast.id = self.id;
                // compare everything now that we have the same ID's
                cast == *self
            }
            // if the rule is invalid, the two are not equal by definition
            Err(_) => false,
        }
    }
}

/// Check that the optional rules in a [`NetworkPolicy`] and a
/// [`NetworkPolicyRequest`] match
macro_rules! rules_opts_match {
    ($policy_rules:expr, $req_rules:expr) => {
        // first compare rules
        match (&$policy_rules, &$req_rules) {
            // they're both None so move on
            (None, None) => (),
            // one is Some and the other is None so they don't match
            (None, Some(_)) | (Some(_), None) => return false,
            // compare all rules if they're both Some
            (Some(pol_rules), Some(req_rules)) => {
                if pol_rules != req_rules {
                    return false;
                }
            }
        }
    };
}

impl PartialEq<NetworkPolicyRequest> for NetworkPolicy {
    /// Checks if all the info in a [`NetworkPolicyRequest`] was set for a [`NetworkPolicy`]
    ///
    /// The only "gotcha" is that the conversion from `NetworkPolicyRuleRaw` to
    /// `NetworkPolicyRule` is fallible, so if any of the raw rules in the request are invalid,
    /// the two are not equal by definition
    ///
    /// # Arguments
    ///
    /// * `req` - The request to compare against
    fn eq(&self, req: &NetworkPolicyRequest) -> bool {
        // check that rules match
        rules_opts_match!(self.ingress, req.ingress);
        rules_opts_match!(self.egress, req.egress);
        // make sure the groups lists are sorted the same for comparison
        let mut policy_groups = self.groups.clone();
        policy_groups.sort_unstable();
        let mut req_groups = req.groups.clone();
        req_groups.sort_unstable();
        // compare fields
        self.name == req.name
            && policy_groups == req_groups
            && self.forced_policy == req.forced_policy
    }
}

impl PartialEq<NetworkPolicyUpdate> for NetworkPolicy {
    /// Verify that all the elements in a [`NetworkPolicyUpdate`] were
    /// applied to a [`NetworkPolicy`]
    fn eq(&self, update: &NetworkPolicyUpdate) -> bool {
        matches_update!(self.name, update.new_name);
        matches_adds!(self.groups, update.add_groups);
        matches_removes!(self.groups, update.remove_groups);
        // check that we set rules to None if we wanted to allow all
        matches_clear!(self.ingress, update.clear_ingress);
        matches_clear!(self.egress, update.clear_egress);
        // check that we cleared rules (empty Vec) if we wanted to deny all
        matches_clear_vec_opt!(self.ingress, update.deny_all_ingress);
        matches_clear_vec_opt!(self.egress, update.deny_all_egress);
        // only check that rules were added if we didn't clear them
        if !update.clear_egress {
            matches_adds_iter!(self.ingress.iter().flatten(), update.add_ingress.iter());
        }
        if !update.clear_egress {
            matches_adds_iter!(self.egress.iter().flatten(), update.add_egress.iter());
        }
        // check that we removed rules
        matches_removes_iter!(
            self.ingress.iter().flatten().map(|rule| &rule.id),
            update.remove_ingress.iter()
        );
        matches_removes_iter!(
            self.egress.iter().flatten().map(|rule| &rule.id),
            update.remove_egress.iter()
        );
        matches_update!(self.forced_policy, update.forced_policy);
        matches_update!(self.default_policy, update.default_policy);
        true
    }
}

/// Compare two serializable values by their JSON representation
///
/// This is used to compare entity metadata whose request and response types are
/// identical but which do not implement [`PartialEq`].
///
/// # Arguments
///
/// * `left` - The left value to compare
/// * `right` - The right value to compare
fn json_eq<L: serde::Serialize, R: serde::Serialize>(left: &L, right: &R) -> bool {
    // serialize both sides and compare their JSON values
    match (serde_json::to_value(left), serde_json::to_value(right)) {
        (Ok(left), Ok(right)) => left == right,
        // if either side failed to serialize then they can't be equal
        _ => false,
    }
}

/// Check that a [`VendorEntity`] corresponds to a [`VendorEntityRequest`]
///
/// # Arguments
///
/// * `resp` - The vendor entity to compare against
/// * `req` - The vendor request to compare against
fn vendor_matches(resp: &VendorEntity, req: &VendorEntityRequest) -> bool {
    // convert the request's country codes into full country objects
    let mut req_countries = BTreeSet::new();
    for code in &req.countries {
        match Country::new(code) {
            Ok(country) => {
                req_countries.insert(country);
            }
            // an invalid country code means the two can't be equal
            Err(_) => return false,
        }
    }
    // compare countries and critical sectors
    resp.countries == req_countries && resp.critical_sectors == req.critical_sectors
}

/// Check that a [`DeviceEntity`] corresponds to a [`DeviceEntityRequest`]
///
/// # Arguments
///
/// * `resp` - The device entity to compare against
/// * `req` - The device request to compare against
fn device_matches(resp: &DeviceEntity, req: &DeviceEntityRequest) -> bool {
    // make sure the simple fields match
    if !matches_vecs_helper(&resp.urls, &req.urls)
        || resp.critical_system != req.critical_system
        || resp.sensitive_location != req.sensitive_location
        || resp.critical_sectors != req.critical_sectors
    {
        return false;
    }
    // the response contains full vendor entities, so compare them by their ids
    let resp_ids: BTreeSet<Uuid> = resp.vendors.iter().map(|vendor| vendor.id).collect();
    let req_ids: BTreeSet<Uuid> = req.vendors.iter().copied().collect();
    resp_ids == req_ids
}

/// Check that a [`CollectionEntity`] corresponds to a [`CollectionEntityRequest`]
///
/// # Arguments
///
/// * `resp` - The collection entity to compare against
/// * `req` - The collection request to compare against
fn collection_matches(resp: &CollectionEntity, req: &CollectionEntityRequest) -> bool {
    // collection kinds don't implement PartialEq so compare their str forms
    if resp.collection_kind.as_ref() != req.collection_kind.as_ref() {
        return false;
    }
    // make sure both sides have the same number of tag keys
    if resp.collection_tags.len() != req.collection_tags.len() {
        return false;
    }
    // make sure every tag key/value in the request is present in the response
    for (key, req_values) in &req.collection_tags {
        match resp.collection_tags.get(key) {
            Some(resp_values) => {
                // compare the two sets of values ignoring ordering
                let req_set: HashSet<&String> = req_values.iter().collect();
                let resp_set: HashSet<&String> = resp_values.iter().collect();
                if req_set != resp_set {
                    return false;
                }
            }
            // the response is missing a tag key from the request
            None => return false,
        }
    }
    // the request stores its flags as options that default to false in the response
    resp.tags_case_insensitive == req.tags_case_insensitive.unwrap_or(false)
        && resp.ignore_groups == req.ignore_groups.unwrap_or(false)
        && resp.start == req.start
        && resp.end == req.end
}

impl PartialEq<EntityRequest> for Entity {
    /// Check if an [`Entity`] corresponds to an [`EntityRequest`]
    ///
    /// # Arguments
    ///
    /// * `req` - The `EntityRequest` to compare against
    fn eq(&self, req: &EntityRequest) -> bool {
        // make sure the shared fields are the same
        same!(self.name, req.name);
        same!(self.kind, req.kind());
        same!(self.description, req.description);
        // the groups may be returned in a different order
        matches_vec!(&self.groups, &req.groups);
        // make sure every tag key/value in the request was applied to the entity
        for (key, values) in &req.tags {
            match self.tags.get(key) {
                Some(value_map) => {
                    for value in values {
                        if !value_map.contains_key(value) {
                            return false;
                        }
                    }
                }
                // the entity is missing a tag key from the request
                None => return false,
            }
        }
        // compare the kind-specific metadata
        match (&self.metadata, &req.metadata) {
            // these kinds carry no comparable metadata beyond their kind
            (EntityMetadata::Other, EntityMetadataRequest::Other)
            | (EntityMetadata::WindowsProcessTree(_), EntityMetadataRequest::WindowsProcessTree) => {
                true
            }
            // these kinds have distinct request/response types so compare them explicitly
            (EntityMetadata::Vendor(resp), EntityMetadataRequest::Vendor(req)) => {
                vendor_matches(resp, req)
            }
            (EntityMetadata::Device(resp), EntityMetadataRequest::Device(req)) => {
                device_matches(resp, req)
            }
            (EntityMetadata::Collection(resp), EntityMetadataRequest::Collection(req)) => {
                collection_matches(resp, req)
            }
            // these kinds share a type between request and response so compare via JSON
            (EntityMetadata::FileSystem(resp), EntityMetadataRequest::FileSystem(req)) => {
                json_eq(resp, req)
            }
            (EntityMetadata::Folder(resp), EntityMetadataRequest::Folder(req)) => json_eq(resp, req),
            (EntityMetadata::WindowsProcess(resp), EntityMetadataRequest::WindowsProcess(req)) => {
                json_eq(resp, req)
            }
            (
                EntityMetadata::NetworkConnection(resp),
                EntityMetadataRequest::NetworkConnection(req),
            ) => json_eq(resp, req),
            (EntityMetadata::PeSection(resp), EntityMetadataRequest::PeSection(req)) => {
                json_eq(resp, req)
            }
            (EntityMetadata::PeImport(resp), EntityMetadataRequest::PeImport(req)) => {
                json_eq(resp, req)
            }
            (EntityMetadata::SigmaRule(resp), EntityMetadataRequest::SigmaRule(req)) => {
                json_eq(resp, req)
            }
            (EntityMetadata::Flag(resp), EntityMetadataRequest::Flag(req)) => json_eq(resp, req),
            (EntityMetadata::Incident(resp), EntityMetadataRequest::Incident(req)) => {
                json_eq(resp, req)
            }
            (
                EntityMetadata::CompiledFunction(resp),
                EntityMetadataRequest::CompiledFunction(req),
            ) => json_eq(resp, req),
            (
                EntityMetadata::DecompiledFunction(resp),
                EntityMetadataRequest::DecompiledFunction(req),
            ) => json_eq(resp, req),
            // any mismatched pairing of kinds means they aren't equal
            _ => false,
        }
    }
}

impl PartialEq<EntityUpdate> for Entity {
    /// Verify that all the elements in an [`EntityUpdate`] were applied to an [`Entity`]
    ///
    /// # Arguments
    ///
    /// * `update` - The `EntityUpdate` to verify was applied
    fn eq(&self, update: &EntityUpdate) -> bool {
        // check that the name was updated if requested
        matches_update!(self.name, update.name);
        // check that the description was cleared or updated as requested
        matches_clear!(self.description, update.clear_description);
        if !update.clear_description {
            matches_update_opt!(self.description, update.description);
        }
        // check that the groups were added and removed as requested
        matches_adds!(self.groups, update.add_groups);
        matches_removes!(self.groups, update.remove_groups);
        // verify any kind-specific metadata update was applied
        if let Some(meta) = &update.metadata {
            match (meta, &self.metadata) {
                (
                    EntityMetadataUpdate::Device {
                        add_urls,
                        remove_urls,
                        critical_system,
                        clear_critical_system,
                        sensitive_location,
                        clear_sensitive_location,
                        add_critical_sectors,
                        remove_critical_sectors,
                    },
                    EntityMetadata::Device(dev),
                ) => {
                    // added urls present, removed urls absent
                    if !add_urls.iter().all(|url| dev.urls.contains(url)) {
                        return false;
                    }
                    if remove_urls.iter().any(|url| dev.urls.contains(url)) {
                        return false;
                    }
                    // critical system flag was cleared or set
                    match (clear_critical_system, critical_system) {
                        (Some(true), _) => {
                            if dev.critical_system.is_some() {
                                return false;
                            }
                        }
                        (_, Some(val)) => {
                            if dev.critical_system != Some(*val) {
                                return false;
                            }
                        }
                        _ => {}
                    }
                    // sensitive location flag was cleared or set
                    match (clear_sensitive_location, sensitive_location) {
                        (Some(true), _) => {
                            if dev.sensitive_location.is_some() {
                                return false;
                            }
                        }
                        (_, Some(val)) => {
                            if dev.sensitive_location != Some(*val) {
                                return false;
                            }
                        }
                        _ => {}
                    }
                    // added sectors present, removed sectors absent
                    if !add_critical_sectors
                        .iter()
                        .all(|sector| dev.critical_sectors.contains(sector))
                    {
                        return false;
                    }
                    if remove_critical_sectors
                        .iter()
                        .any(|sector| dev.critical_sectors.contains(sector))
                    {
                        return false;
                    }
                }
                (
                    EntityMetadataUpdate::Vendor {
                        add_countries,
                        remove_countries,
                        add_critical_sectors,
                        remove_critical_sectors,
                    },
                    EntityMetadata::Vendor(vendor),
                ) => {
                    // added countries (alpha-2 codes) present
                    for code in add_countries {
                        match Country::new(code) {
                            Ok(country) => {
                                if !vendor.countries.contains(&country) {
                                    return false;
                                }
                            }
                            Err(_) => return false,
                        }
                    }
                    // removed countries absent
                    for code in remove_countries {
                        if let Ok(country) = Country::new(code) {
                            if vendor.countries.contains(&country) {
                                return false;
                            }
                        }
                    }
                    // added sectors present, removed sectors absent
                    if !add_critical_sectors
                        .iter()
                        .all(|sector| vendor.critical_sectors.contains(sector))
                    {
                        return false;
                    }
                    if remove_critical_sectors
                        .iter()
                        .any(|sector| vendor.critical_sectors.contains(sector))
                    {
                        return false;
                    }
                }
                (
                    EntityMetadataUpdate::Collection {
                        add_collection_tags,
                        delete_collection_tags,
                        tags_case_insensitive,
                        ignore_groups,
                        start,
                        end,
                        clear_start,
                        clear_end,
                    },
                    EntityMetadata::Collection(col),
                ) => {
                    // added tag values present under their key
                    for (key, values) in add_collection_tags {
                        match col.collection_tags.get(key) {
                            Some(existing) => {
                                if !values.iter().all(|val| existing.contains(val)) {
                                    return false;
                                }
                            }
                            None => return false,
                        }
                    }
                    // deleted tag values absent
                    for (key, values) in delete_collection_tags {
                        if let Some(existing) = col.collection_tags.get(key) {
                            if values.iter().any(|val| existing.contains(val)) {
                                return false;
                            }
                        }
                    }
                    if let Some(val) = tags_case_insensitive {
                        if col.tags_case_insensitive != *val {
                            return false;
                        }
                    }
                    if let Some(val) = ignore_groups {
                        if col.ignore_groups != *val {
                            return false;
                        }
                    }
                    // start was cleared or set
                    match (clear_start, start) {
                        (Some(true), _) => {
                            if col.start.is_some() {
                                return false;
                            }
                        }
                        (_, Some(ts)) => {
                            if col.start != Some(*ts) {
                                return false;
                            }
                        }
                        _ => {}
                    }
                    // end was cleared or set
                    match (clear_end, end) {
                        (Some(true), _) => {
                            if col.end.is_some() {
                                return false;
                            }
                        }
                        (_, Some(ts)) => {
                            if col.end != Some(*ts) {
                                return false;
                            }
                        }
                        _ => {}
                    }
                }
                (
                    EntityMetadataUpdate::FileSystem {
                        add_tools,
                        remove_tools,
                    },
                    EntityMetadata::FileSystem(fs),
                ) => {
                    if !add_tools.iter().all(|tool| fs.tools.contains(tool)) {
                        return false;
                    }
                    if remove_tools.iter().any(|tool| fs.tools.contains(tool)) {
                        return false;
                    }
                }
                (
                    EntityMetadataUpdate::WindowsProcessTree {
                        add_tools,
                        remove_tools,
                    },
                    EntityMetadata::WindowsProcessTree(tree),
                ) => {
                    if !add_tools.iter().all(|tool| tree.tools.contains(tool)) {
                        return false;
                    }
                    if remove_tools.iter().any(|tool| tree.tools.contains(tool)) {
                        return false;
                    }
                }
                (
                    EntityMetadataUpdate::WindowsProcess {
                        name,
                        image_path,
                        command,
                        offset,
                        threads,
                        handles,
                        is_wow64,
                        session_id,
                        create_time,
                        exit_time,
                    },
                    EntityMetadata::WindowsProcess(win_proc),
                ) => {
                    if let Some(val) = name {
                        if win_proc.name.as_ref() != Some(val) {
                            return false;
                        }
                    }
                    if let Some(val) = image_path {
                        if win_proc.image_path.as_ref() != Some(val) {
                            return false;
                        }
                    }
                    if let Some(val) = command {
                        if win_proc.command.as_ref() != Some(val) {
                            return false;
                        }
                    }
                    if let Some(val) = offset {
                        if win_proc.offset != Some(*val) {
                            return false;
                        }
                    }
                    if let Some(val) = threads {
                        if win_proc.threads != Some(*val) {
                            return false;
                        }
                    }
                    if let Some(val) = handles {
                        if win_proc.handles != Some(*val) {
                            return false;
                        }
                    }
                    if let Some(val) = is_wow64 {
                        if win_proc.is_wow64 != Some(*val) {
                            return false;
                        }
                    }
                    if let Some(val) = session_id {
                        if win_proc.session_id != Some(*val) {
                            return false;
                        }
                    }
                    if let Some(val) = create_time {
                        if win_proc.create_time != Some(*val) {
                            return false;
                        }
                    }
                    if let Some(val) = exit_time {
                        if win_proc.exit_time != Some(*val) {
                            return false;
                        }
                    }
                }
                (
                    EntityMetadataUpdate::NetworkConnection {
                        protocol,
                        source,
                        source_port,
                        destination,
                        destination_port,
                        state,
                        pid,
                        process,
                        create_time,
                    },
                    EntityMetadata::NetworkConnection(conn),
                ) => {
                    // protocol/state have no PartialEq, so compare their display form
                    if let Some(val) = protocol {
                        if conn.protocol.as_ref().map(ToString::to_string)
                            != Some(val.to_string())
                        {
                            return false;
                        }
                    }
                    if let Some(val) = source {
                        if conn.source != *val {
                            return false;
                        }
                    }
                    if let Some(val) = source_port {
                        if conn.source_port != Some(*val) {
                            return false;
                        }
                    }
                    if let Some(val) = destination {
                        if conn.destination != *val {
                            return false;
                        }
                    }
                    if let Some(val) = destination_port {
                        if conn.destination_port != *val {
                            return false;
                        }
                    }
                    if let Some(val) = state {
                        if conn.state.as_ref().map(ToString::to_string) != Some(val.to_string()) {
                            return false;
                        }
                    }
                    if let Some(val) = pid {
                        if conn.pid != Some(*val) {
                            return false;
                        }
                    }
                    if let Some(val) = process {
                        if conn.process.as_ref() != Some(val) {
                            return false;
                        }
                    }
                    if let Some(val) = create_time {
                        if conn.create_time != Some(*val) {
                            return false;
                        }
                    }
                }
                (
                    EntityMetadataUpdate::SigmaRule {
                        sigma_rule,
                        score,
                        add_applies_to,
                        remove_applies_to,
                        add_actions,
                        remove_actions: _,
                    },
                    EntityMetadata::SigmaRule(rule),
                ) => {
                    if let Some(val) = sigma_rule {
                        if &rule.rule != val {
                            return false;
                        }
                    }
                    if let Some(val) = score {
                        if rule.score != *val {
                            return false;
                        }
                    }
                    if !add_applies_to
                        .iter()
                        .all(|applies| rule.applies_to.contains(applies))
                    {
                        return false;
                    }
                    if remove_applies_to
                        .iter()
                        .any(|applies| rule.applies_to.contains(applies))
                    {
                        return false;
                    }
                    // actions have no PartialEq; they were created empty so must json-match the adds
                    if !add_actions.is_empty() && !json_eq(&rule.actions, add_actions) {
                        return false;
                    }
                }
                (
                    EntityMetadataUpdate::Flag {
                        suspicion,
                        confidence,
                        reasoning,
                        content,
                    },
                    EntityMetadata::Flag(flag),
                ) => {
                    if let Some(val) = suspicion {
                        if flag.suspicion != *val {
                            return false;
                        }
                    }
                    // confidence has no PartialEq, so compare its display form
                    if let Some(val) = confidence {
                        if flag.confidence.to_string() != val.to_string() {
                            return false;
                        }
                    }
                    if let Some(val) = reasoning {
                        if &flag.reasoning != val {
                            return false;
                        }
                    }
                    if let Some(val) = content {
                        if flag.content.as_ref() != Some(val) {
                            return false;
                        }
                    }
                }
                (
                    EntityMetadataUpdate::Incident {
                        cover_term,
                        add_mission_teams,
                        remove_mission_teams,
                        add_networks,
                        remove_networks,
                        add_machines,
                        remove_machines,
                        add_locations,
                        remove_locations,
                    },
                    EntityMetadata::Incident(incident),
                ) => {
                    if let Some(val) = cover_term {
                        if incident.cover_term.as_ref() != Some(val) {
                            return false;
                        }
                    }
                    if !add_mission_teams
                        .iter()
                        .all(|team| incident.mission_teams.contains(team))
                        || remove_mission_teams
                            .iter()
                            .any(|team| incident.mission_teams.contains(team))
                    {
                        return false;
                    }
                    if !add_networks
                        .iter()
                        .all(|net| incident.networks.contains(net))
                        || remove_networks
                            .iter()
                            .any(|net| incident.networks.contains(net))
                    {
                        return false;
                    }
                    if !add_machines
                        .iter()
                        .all(|machine| incident.machines.contains(machine))
                        || remove_machines
                            .iter()
                            .any(|machine| incident.machines.contains(machine))
                    {
                        return false;
                    }
                    if !add_locations
                        .iter()
                        .all(|loc| incident.locations.contains(loc))
                        || remove_locations
                            .iter()
                            .any(|loc| incident.locations.contains(loc))
                    {
                        return false;
                    }
                }
                (
                    EntityMetadataUpdate::CompiledFunction {
                        function_address,
                        disassembly,
                    },
                    EntityMetadata::CompiledFunction(func),
                ) => {
                    if let Some(val) = function_address {
                        if func.address != *val {
                            return false;
                        }
                    }
                    // disassembly has no PartialEq; it is fully replaced so must json-match
                    if !disassembly.is_empty() && !json_eq(&func.disassembly, disassembly) {
                        return false;
                    }
                }
                (
                    EntityMetadataUpdate::DecompiledFunction {
                        function_address,
                        decompilation_content,
                        add_tools,
                        remove_tools,
                    },
                    EntityMetadata::DecompiledFunction(decomp),
                ) => {
                    if let Some(val) = function_address {
                        if decomp.address != *val {
                            return false;
                        }
                    }
                    if let Some(val) = decompilation_content {
                        if &decomp.content != val {
                            return false;
                        }
                    }
                    if !add_tools.iter().all(|tool| decomp.tools.contains(tool)) {
                        return false;
                    }
                    if remove_tools.iter().any(|tool| decomp.tools.contains(tool)) {
                        return false;
                    }
                }
                (
                    EntityMetadataUpdate::PeSection {
                        md5,
                        raw_size,
                        virtual_size,
                        entropy,
                    },
                    EntityMetadata::PeSection(section),
                ) => {
                    if let Some(val) = md5 {
                        if section.md5.as_ref() != Some(val) {
                            return false;
                        }
                    }
                    if let Some(val) = raw_size {
                        if section.raw_size != Some(*val) {
                            return false;
                        }
                    }
                    if let Some(val) = virtual_size {
                        if section.virtual_size != Some(*val) {
                            return false;
                        }
                    }
                    if let Some(val) = entropy {
                        if section.entropy != Some(*val) {
                            return false;
                        }
                    }
                }
                (
                    EntityMetadataUpdate::PeImport { functions },
                    EntityMetadata::PeImport(import),
                ) => {
                    // imported functions are fully replaced
                    if !functions.is_empty() && &import.functions != functions {
                        return false;
                    }
                }
                // a metadata update for the wrong kind means it wasn't applied
                _ => return false,
            }
        }
        true
    }
}
