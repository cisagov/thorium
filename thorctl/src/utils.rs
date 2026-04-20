//! A module containing various utility functions for use in multiple handlers

use colored::Colorize;
use data_encoding::HEXLOWER;
use regex::{Regex, RegexSet};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use thorium::Keys;
use thorium::{Error, Thorium};
use tokio::io::{AsyncReadExt, BufReader};

use crate::{Args, CtlConf};

/// The per-request limit used to drain a cursor in full ("list everything")
///
/// The list APIs are still paged underneath this cap; it's just a ceiling chosen
/// far above any realistic group's image/pipeline count. A group can't plausibly
/// hold this many resources, so it stands in for "no limit" without a dedicated
/// unbounded-cursor API.
pub const LIST_ALL_LIMIT: u64 = 1_000_000;

/// Serialize a value to pretty JSON with object keys sorted
///
/// `HashMap`/`HashSet` fields (e.g. an image's `env`, a pipeline's `triggers`)
/// serialize in a random, per-run order, which makes otherwise-identical configs
/// byte-differ between runs. Routing through `serde_json::Value` (a sorted
/// `BTreeMap`) yields a canonical, order-stable form so equality and diff
/// comparisons treat reordered maps as identical.
///
/// # Arguments
///
/// * `value` - The value to serialize
pub fn canonical_json<T: Serialize>(value: &T) -> Result<String, Error> {
    let sorted = serde_json::to_value(value)
        .map_err(|err| Error::new(format!("Failed to serialize to JSON: {err}")))?;
    serde_json::to_string_pretty(&sorted)
        .map_err(|err| Error::new(format!("Failed to serialize to JSON: {err}")))
}

/// Serialize a value to YAML with mapping keys sorted
///
/// See [`canonical_json`] — this is the YAML form, used for the toolbox diff and
/// the merge editor so reordered maps don't show as spurious changes.
///
/// # Arguments
///
/// * `value` - The value to serialize
pub fn canonical_yaml<T: Serialize>(value: &T) -> Result<String, Error> {
    let sorted = serde_json::to_value(value)
        .map_err(|err| Error::new(format!("Failed to serialize to YAML: {err}")))?;
    serde_norway::to_string(&sorted)
        .map_err(|err| Error::new(format!("Failed to serialize to YAML: {err}")))
}

/// Serialize a value to YAML with a curated top-level key order
///
/// Top-level keys named in `order` are emitted first, in that order; any remaining
/// keys follow in sorted order (so a newly added field is never dropped — it just
/// lands at the end). Each ordered name matches either the plain key or its
/// static-marked `*key*` form, so this works for both the `Mergeable*` editor view
/// (which marks uneditable fields like `*name*`) and the raw config JSON. Nested
/// maps stay sorted (canonical), keeping editor/merge output deterministic.
///
/// # Arguments
///
/// * `value` - The value to serialize
/// * `order` - The curated top-level key order (plain names; `*name*` also matches)
pub fn curated_yaml<T: Serialize>(value: &T, order: &[&str]) -> Result<String, Error> {
    let json = serde_json::to_value(value)
        .map_err(|err| Error::new(format!("Failed to serialize to YAML: {err}")))?;
    // only objects have a key order to curate; anything else falls back to canonical
    let serde_json::Value::Object(map) = json else {
        return canonical_yaml(value);
    };
    let mut out = serde_norway::Mapping::new();
    for (key, v) in curated_entries(map, order) {
        let yaml_value = serde_norway::to_value(&v)
            .map_err(|err| Error::new(format!("Failed to serialize to YAML: {err}")))?;
        out.insert(serde_norway::Value::String(key), yaml_value);
    }
    serde_norway::to_string(&out)
        .map_err(|err| Error::new(format!("Failed to serialize to YAML: {err}")))
}

/// Serialize a value to pretty JSON with a curated top-level key order
///
/// The JSON twin of [`curated_yaml`]: top-level keys in the curated order, then any
/// remaining keys sorted at the end (never dropped); nested objects stay sorted. Used
/// to write scaffolded `<name>.json` configs in an edit-friendly order rather than
/// alphabetical.
///
/// # Arguments
///
/// * `value` - The value to serialize
/// * `order` - The curated top-level key order (plain names; `*name*` also matches)
pub fn curated_json<T: Serialize>(value: &T, order: &[&str]) -> Result<String, Error> {
    use serde::ser::{SerializeMap, Serializer};
    let json = serde_json::to_value(value)
        .map_err(|err| Error::new(format!("Failed to serialize to JSON: {err}")))?;
    // only objects have a key order to curate; anything else serializes as-is
    let serde_json::Value::Object(map) = json else {
        return serde_json::to_string_pretty(value)
            .map_err(|err| Error::new(format!("Failed to serialize to JSON: {err}")));
    };
    let entries = curated_entries(map, order);
    // drive the serializer's map directly so the (insertion) order we feed is the
    // emitted order — `serde_json::Value`'s own `BTreeMap` would re-sort it
    let mut ser = serde_json::Serializer::pretty(Vec::new());
    let mut map_ser = ser
        .serialize_map(Some(entries.len()))
        .map_err(|err| Error::new(format!("Failed to serialize to JSON: {err}")))?;
    for (key, v) in &entries {
        map_ser
            .serialize_entry(key, v)
            .map_err(|err| Error::new(format!("Failed to serialize to JSON: {err}")))?;
    }
    SerializeMap::end(map_ser)
        .map_err(|err| Error::new(format!("Failed to serialize to JSON: {err}")))?;
    String::from_utf8(ser.into_inner())
        .map_err(|err| Error::new(format!("Failed to serialize to JSON: {err}")))
}

/// Reorder a JSON object's top-level entries into a curated order: keys named in
/// `order` first (matching either `key` or its static-marked `*key*` form), then any
/// remaining keys in sorted order. Shared by [`curated_yaml`] and [`curated_json`] so
/// the ordering rule lives in one place. No key is ever dropped.
///
/// # Arguments
///
/// * `map` - The object's entries (a `serde_json` `BTreeMap`, so leftovers stay sorted)
/// * `order` - The curated top-level key order (plain names; `*name*` also matches)
fn curated_entries(
    mut map: serde_json::Map<String, serde_json::Value>,
    order: &[&str],
) -> Vec<(String, serde_json::Value)> {
    let mut entries = Vec::with_capacity(map.len());
    // pull the curated keys out first, matching either `key` or the `*key*` static form
    for name in order {
        for candidate in [name.to_string(), format!("*{name}*")] {
            if let Some(value) = map.remove(&candidate) {
                entries.push((candidate, value));
            }
        }
    }
    // append whatever's left; a `serde_json::Map` iterates sorted, so unlisted keys
    // land at the end in a stable order
    entries.extend(map);
    entries
}

pub mod banner;
pub mod diff;
pub mod fs;
pub mod groups;
pub mod images;
pub mod notifications;
pub mod pipelines;
pub mod reactions;
pub mod render;
pub mod repos;

/// Get a Thorium client or setup keys
pub async fn get_client(args: &Args) -> Result<(CtlConf, Thorium), Error> {
    let (config, thorium) = match &args.keys {
        Some(keys_path) => {
            // parse the keys from the file
            let keys = Keys::from_path(keys_path)?;
            // build our Thorium client based on our config
            let thorium = Thorium::from_keys(keys.clone()).await?;
            // build a base ctl conf containing the keys
            let config = CtlConf::new(keys);
            (config, thorium)
        }
        None => {
            // load ctl conf
            let config = CtlConf::from_path(&args.config)?;
            // build our Thorium client based on our config
            let thorium = Thorium::from_ctl_conf(config.clone()).await?;
            (config, thorium)
        }
    };
    Ok((config, thorium))
}

/// Get the sha256 for a file
pub async fn sha256<P: AsRef<Path>>(path: P) -> Result<String, Error> {
    // get buffered reader for this file
    let file = tokio::fs::File::open(path).await?;
    let mut reader = BufReader::new(file);
    // read this file into a local buffer and hash it
    let mut sha256 = Sha256::new();
    let mut buff = [0; 2048];
    loop {
        // read in 2048 bytes and count how many are read
        let count = reader.read(&mut buff[..]).await?;
        // if we read in no bytes then we have read our entire file
        if count == 0 {
            break;
        }
        // update our hashers with our newly read data
        sha256.update(&buff[..count]);
    }
    // build a digest for this
    let sha256 = HEXLOWER.encode(&sha256.finalize());
    Ok(sha256)
}

/// Returns true if the haystack matches user filters
///
/// If filters are provided, returns true if the haystack matches at least one of the
/// regexes in the set. If skip filters are provided, returns true if the haystack
/// doesn't match any of the regexes in the set. If both are provided, returns true
/// if the haystack both matches at least one filter in the filter set and doesn't
/// match any filters in the skip set.
///
/// Otherwise, returns false.
///
/// # Arguments
///
/// * `haystack` - The haystack we're checking
/// * `filter` - A set of regular expressions to use to determine which files to include
/// * `skip` - A set of regular expressions to use to determine which files to skip
pub fn filter_str(haystack: &str, filter: &RegexSet, skip: &RegexSet) -> bool {
    // get the path's filename
    match (filter.is_empty(), skip.is_empty()) {
        // needs to match the filter and not the skip if both are given
        (false, false) => filter.is_match(haystack) && !skip.is_match(haystack),
        (true, false) => !skip.is_match(haystack),
        (false, true) => filter.is_match(haystack),
        (true, true) => true,
    }
}

/// Log any errors and return
#[doc(hidden)]
#[macro_export]
macro_rules! check {
    ($self:expr, $result:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => {
                // log this error
                $self.bar.error(error.to_string());
                // return early
                return;
            }
        }
    };
    ($self:expr, $result:expr, $path:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => {
                // log this error
                $self.bar.error(error.to_string());
                // check if our path exists
                if let Ok(true) = tokio::fs::try_exists($path).await {
                    // clean up this repos dir
                    if let Err(error) = tokio::fs::remove_dir_all($path).await {
                        // log this io error
                        $self.bar.error(error.to_string());
                    }
                }
                // return early
                return;
            }
        }
    };
}

/// Print a warning message that Thorctl is possibly insecure as well as a command to run
/// to disable the setting that caused the warning message
macro_rules! print_warning {
    ($msg:expr, $api:expr, $cmd:expr) => {
        println!(
            "{}: Thorctl is currently set to {} when connecting \
            to Thorium. Only continue if you 100% trust the instance at '{}'.\n\
            \n    \
            Note: You can avoid this error message in the future by either running `{}` \
            or by disabling this warning message altogether with `{}`\n",
            "WARNING".bright_yellow(),
            $msg.bright_red(),
            $api.blue(),
            $cmd.green(),
            "thorctl config --skip-insecure-warning=true".green()
        );
    };
}

/// Print a warning message if any of the insecure settings are set
///
/// # Arguments
///
/// * `api` - The API we're connecting to
/// * `invalid_certs` - Accept invalid certs
/// * `invalid_hostnames` - Accept invalid hostnames
/// * `certificate_authorities` - A list of certificate authorities to implicitly trust
pub fn warn_insecure(
    api: &str,
    invalid_certs: bool,
    invalid_hostnames: bool,
    certificate_authorities: &[PathBuf],
) -> Result<(), Error> {
    // check possibly insecure settings in order of most to least severe
    if invalid_certs {
        print_warning!(
            "skip all certificate validation",
            api,
            "thorctl config --invalid-certs=false"
        );
    } else if invalid_hostnames {
        print_warning!(
            "skip hostname validation",
            api,
            "thorctl config --invalid-hostnames=false"
        );
    } else if !certificate_authorities.is_empty() {
        print_warning!(
            format!(
                "implicitly trust certificate authorities '{:?}'",
                certificate_authorities
            ),
            api,
            "thorctl config --clear-certificate-authorities"
        );
    } else {
        // return immediately if none of the insecure options are set
        return Ok(());
    }
    // ask the user for permission to update Thorctl
    let response = dialoguer::Confirm::new()
        .with_prompt("Continue?:")
        .interact()?;
    if !response {
        // inform the user Thorctl will exit then exit
        println!("Exiting...");
        std::process::exit(0);
    }
    Ok(())
}

/// Ensure a confirmation prompt can actually run before we try to show one
///
/// The destructive commands (`images`/`pipelines delete`, `toolbox remove`) confirm
/// before acting. Without a terminal (CI, pipes) `dialoguer` can't read an answer and
/// surfaces an opaque IO error, so check up front and fail with a message that names
/// the flag which skips the prompt — keeping non-interactive runs fail-closed and
/// legible instead of dumping a raw dialoguer error.
///
/// # Arguments
///
/// * `skip_flag` - The flag that bypasses the prompt (e.g. "--skip-confirm (-y)")
pub fn require_confirm_terminal(skip_flag: &str) -> Result<(), Error> {
    if std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        Ok(())
    } else {
        Err(Error::new(format!(
            "No terminal available to confirm this action; pass {skip_flag} to proceed non-interactively"
        )))
    }
}

/// Print an insecure warning message if a [`CtlConf`] is configured for
/// insecure connections
///
/// # Arguments
///
/// * `conf` - The [`CtlConf`] set when running the command
pub fn warn_insecure_conf(conf: &CtlConf) -> Result<(), Error> {
    warn_insecure(
        &conf.keys.api,
        conf.client.invalid_certs,
        conf.client.invalid_hostnames,
        &conf.client.certificate_authorities,
    )
}

/// Matches the first SI/binary unit prefix in a storage value, used to split the
/// numeric amount from its unit. Compiled once: the same pattern is reused on
/// every call, so a per-call `Regex::new` would re-pay the compile cost.
static UNIT_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[KMGTPE]").expect("storage unit regex is valid"));

/// Converts a size to bytes
///
/// # Arguments
///
/// * `raw` - A raw storage value
pub fn convert_size_to_bytes(raw: &str) -> Result<u64, Error> {
    // a bare number is already a byte count
    if let Ok(bytes) = raw.parse::<u64>() {
        return Ok(bytes);
    }
    // otherwise locate the unit prefix that follows the numeric amount
    let reg = match UNIT_REGEX.find(raw) {
        Some(reg) => reg,
        None => return Err(Error::new(format!("Failed to parse unit: {raw}"))),
    };
    // split raw based on where unit was found
    let (amt, unit) = raw.split_at(reg.start());
    // cast amt to u64
    let amt = match amt.parse::<u64>() {
        Ok(amt) => amt,
        Err(error) => {
            return Err(Error::new(format!(
                "Failed to parse amount: {raw} - {error:#?}"
            )));
        }
    };
    // the multiplier for each supported decimal (K/M/…) and binary (Ki/Mi/…) unit
    let multiplier: u64 = match unit {
        "K" => 1000,
        "M" => 1_000_000,
        "G" => 1_000_000_000,
        "T" => 1_000_000_000_000,
        "P" => 1_000_000_000_000_000,
        "E" => 1_000_000_000_000_000_000,
        "Ki" => 1024,
        "Mi" => 1_048_576,
        "Gi" => 1_073_741_824,
        "Ti" => 1_099_511_627_776,
        "Pi" => 1_125_899_906_842_624,
        "Ei" => 1_152_921_504_606_846_976,
        _ => return Err(Error::new(format!("Failed to parse storage value: {raw}"))),
    };
    // use checked arithmetic so a huge value reports an error instead of wrapping
    amt.checked_mul(multiplier)
        .ok_or_else(|| Error::new(format!("Storage value '{raw}' overflows u64")))
}

/// Extract a hostname from a url
///
/// # Arguments
///
/// * `url` - The url to get a base from
pub fn get_hostname(url: &str) -> Result<&str, Error> {
    // if this url has a base then strip it out
    let without_base = if url.contains("://") {
        // split on "://" to skip the base
        match url.split("://").nth(1) {
            Some(without_base) => without_base,
            None => return Err(Error::new(format!("Failed to get hostname for {url}"))),
        }
    } else {
        url
    };
    // get just the hostname from our trimmed url
    match without_base.split('/').next() {
        Some(hostname) => Ok(hostname),
        None => Err(Error::new(format!("Failed to get hostname for {url}"))),
    }
}

/// Return a descriptive error that the function requires admin access if we get
/// a 401, otherwise just return the error
#[macro_export]
macro_rules! err_not_admin {
    ($func:expr) => {
        if let Err(err) = $func {
            if err
                .status()
                .is_some_and(|status| status == http::StatusCode::UNAUTHORIZED)
            {
                return Err(Error::new("You must be an admin to perform this function!"));
            }
            // just return the error if not a 401
            return Err(err);
        }
    };
    ($func:expr, $msg:expr) => {
        if let Err(err) = $func {
            if err
                .status()
                .is_some_and(|status| status == http::StatusCode::UNAUTHORIZED)
            {
                return Err(Error::new(format!("You must be an admin to {}!", $msg)));
            }
            // just return the error if not a 401
            return Err(err);
        }
    };
}

#[cfg(test)]
mod tests {

    use regex::RegexSet;

    use super::filter_str;

    /// curated_yaml emits curated keys first (matching `key` or `*key*`), then every
    /// remaining key sorted at the end (never dropped), with nested maps sorted
    #[test]
    fn curated_yaml_orders_and_keeps_all_keys() {
        let value = serde_json::json!({
            "zeta": 1,
            "name": "n",
            "*creator*": "c",
            "alpha": 2,
            "scaler": "K8s",
            "nested": { "b": 1, "a": 2 },
        });
        let yaml = super::curated_yaml(&value, &["name", "scaler", "creator"]).unwrap();
        // top-level (column-0) keys, in emitted order (YAML quotes `*creator*`, so
        // strip surrounding quotes before comparing)
        let keys: Vec<String> = yaml
            .lines()
            .filter(|l| !l.starts_with(char::is_whitespace) && l.contains(':'))
            .map(|l| {
                l.split(':')
                    .next()
                    .unwrap()
                    .trim()
                    .trim_matches(['\'', '"'])
                    .to_string()
            })
            .collect();
        // curated first (creator matched via *creator*), then unlisted keys sorted
        assert_eq!(keys, ["name", "scaler", "*creator*", "alpha", "nested", "zeta"]);
        // nested maps stay sorted (a before b)
        assert!(yaml.find("a: 2").unwrap() < yaml.find("b: 1").unwrap());
    }

    /// curated_json mirrors curated_yaml's ordering: curated keys first (matching
    /// `key`/`*key*`), remaining keys sorted at the end (never dropped), nested sorted
    #[test]
    fn curated_json_orders_and_keeps_all_keys() {
        let value = serde_json::json!({
            "zeta": 1,
            "name": "n",
            "*creator*": "c",
            "alpha": 2,
            "scaler": "K8s",
            "nested": { "b": 1, "a": 2 },
        });
        let json = super::curated_json(&value, &["name", "scaler", "creator"]).unwrap();
        // pretty JSON indents top-level keys with exactly two spaces; nested keys get
        // four, so this picks out only the top-level keys in emitted order
        let keys: Vec<String> = json
            .lines()
            .filter(|l| l.starts_with("  \""))
            .filter_map(|l| l.trim().split('"').nth(1).map(String::from))
            .collect();
        assert_eq!(keys, ["name", "scaler", "*creator*", "alpha", "nested", "zeta"]);
        // nested objects stay sorted (a before b)
        assert!(json.find("\"a\"").unwrap() < json.find("\"b\"").unwrap());
    }

    /// With no filter or skip patterns, every string is accepted
    #[test]
    fn test_filter_str_no_filters() {
        let filter = RegexSet::empty();
        let skip = RegexSet::empty();
        // any str should be accepted
        assert!(filter_str("file.txt", &filter, &skip,));
        assert!(filter_str("any/file.txt", &filter, &skip,));
    }

    /// A string is accepted only when it matches a filter pattern and no skip
    /// pattern
    #[test]
    fn test_filter_str_with_filter_and_skip() {
        let filter = RegexSet::new([r".*\.txt$", r".*include.*"]).unwrap();
        let skip = RegexSet::new([r".*ignore.*"]).unwrap();
        // matches filter, not skip -> should succeed
        assert!(filter_str("notes.txt", &filter, &skip,));
        // matches filter but also skip -> should fail
        assert!(!filter_str("include_ignore.txt", &filter, &skip,));
        // does not match filter -> should fail
        assert!(!filter_str("image.png", &filter, &skip,));
    }
}
