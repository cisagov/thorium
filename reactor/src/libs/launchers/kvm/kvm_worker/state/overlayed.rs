//! The logic for a `KvmWorker` in the [`Overlayed`] state

use std::path::{Path, PathBuf};

use thorium::models::{Image, Worker};
use tracing::instrument;
use virt::domain::Domain;

use super::{Defined, KvmWorkerState, Overlayed};
use crate::{
    args::Kvm,
    libs::{
        Error,
        launchers::kvm::{WORKER_DOMAIN_NAME_PREFIX, virt_async::LibvirtClient},
    },
};

impl Overlayed {
    /// Define the worker, defining its domain and returning a handle to it
    ///
    /// # Arguments
    ///
    /// * `args` - The KVM-specific args passed to the reactor
    /// * `client` - A libvirt client
    /// * `worker` - The worker to define a domain for
    ///
    /// # Resource Allocation
    /// - CPU: Always rounds UP from mCPU to vCPU (minimum 1)
    /// - Memory: Uses MiB directly without conversion
    #[instrument(skip_all, fields(worker = self.worker))]
    pub async fn define(
        self,
        client: &LibvirtClient,
        args: &Kvm,
        worker: &Worker,
        image: &Image,
    ) -> Result<(Defined, Domain), Error> {
        // Validate worker resources
        if worker.resources.cpu == 0 {
            return Err(Error::new("CPU resources cannot be zero"));
        }
        if worker.resources.memory == 0 {
            return Err(Error::new("Memory resources cannot be zero"));
        }
        if worker.resources.memory < 512 {
            return Err(Error::new("Memory must be at least 512 MiB"));
        }
        let domain_name = self.domain_name();
        // calculate cpu to allocate; the actual vCPU allocated may be higher than requested
        // mCPU due to rounding up; we always allocate at least 1 CPU
        let vcpu = std::cmp::max(1, worker.resources.cpu.div_ceil(1000));
        let vcpu_str = vcpu.to_string();
        let overlay_path = self.overlay_path.to_string_lossy();
        // read the golden xml file
        let golden_xml = Self::find_golden_xml(&args.golden_dir, image).await?;
        let golden_xml_str = tokio::fs::read_to_string(&golden_xml)
            .await
            .map_err(|err| {
                Error::with_context(
                    format!("Error reading golden XML file '{}'", golden_xml.display()),
                    err,
                )
            })?;
        // convert our memory to whatever units we detect, defaulting to KiB
        let memory = detect_mem_units(&golden_xml_str, worker.resources.memory)
            .unwrap_or(worker.resources.memory * 1024);
        let memory_str = memory.to_string();
        let replacements = [
            ("{NAME}", domain_name.as_str()),
            ("{CPU}", vcpu_str.as_str()),
            ("{MEMORY}", memory_str.as_str()),
            ("{FILE}", &overlay_path),
        ];
        let new_xml_str = apply_replacements(&golden_xml_str, &replacements).map_err(|err| {
            Error::with_context(
                format!("Malformed worker XML '{}'", golden_xml.display()),
                err,
            )
        })?;
        let domain = client
            .with_conn(move |conn| Domain::define_xml(conn, &new_xml_str))
            .await?;
        let defined = self.next_state(domain_name);
        Ok((defined, domain))
    }

    /// Proceed to the next state as if we actually defined the worker
    pub fn mock_define(self) -> Defined {
        // presume the domain name was defined correctly
        let domain_name = self.domain_name();
        self.next_state(domain_name)
    }

    /// Construct a domain name for a this worker
    fn domain_name(&self) -> String {
        // the domain name is the Thorium management prefix prepended to the worker name
        format!("{}{}", WORKER_DOMAIN_NAME_PREFIX, self.worker)
    }

    /// Find the golden XML definition for a given worker
    ///
    /// # Arguments
    ///
    /// * `golden_base` - The path to the base golden directory
    /// * `image` - The image information
    #[instrument(skip_all)]
    async fn find_golden_xml(golden_dir: &Path, image: &Image) -> Result<PathBuf, Error> {
        let mut golden_xml_path = golden_dir.to_path_buf();
        golden_xml_path.push(&image.group);
        golden_xml_path.push(&image.name);
        golden_xml_path.push(&image.name);
        golden_xml_path.set_extension("xml");
        // make sure the file exists
        if !tokio::fs::try_exists(&golden_xml_path).await? {
            return Err(Error::new(format!(
                "Failed to find golden XML at path '{}'",
                golden_xml_path.display()
            )));
        }
        Ok(golden_xml_path)
    }
}

/// Attempt to detect the memory units from the given XML and return the memory allocation
/// converted based on those units or `None` if we failed to detect them
///
/// Based on [libvirt's memory allocation specification](<https://libvirt.org/formatdomain.html#memory-allocation>)
///
/// # Arguments
///
/// * `xml` - The XML string to search for units in
/// * `memory` - The memory allocation in MiB as reported by the Thorium API
fn detect_mem_units(xml: &str, memory: u64) -> Option<u64> {
    const BYTES_IN_MIB: u64 = 1 << 20;
    let memory_needle = "<memory";
    let unit_needle = "unit=";
    // find the memory spec and the units if it was specified
    let memory_index = xml.find(memory_needle)?;
    let unit_index = memory_index + xml[memory_index..].find(unit_needle)?;
    // find the opening quote character (single quote or double quote)
    let quote_char = xml[unit_index..].chars().nth(unit_needle.len())?;
    // get the start index of units, skipping past the unit needle and the starting quote (' or ")
    let unit_start = unit_index + unit_needle.len() + 1;
    // get the end of the units by finding the closing quote character
    let unit_end = unit_start + xml[unit_start..].find(quote_char)?;
    match xml[unit_start..unit_end].to_lowercase().as_str() {
        "b" => Some(memory * BYTES_IN_MIB),
        "kb" => Some(memory * BYTES_IN_MIB / 1000),
        "k" | "kib" => Some(memory * 1024),
        "m" | "mib" => Some(memory),
        "mb" => Some(memory * BYTES_IN_MIB / 1_000_000),
        "gb" => Some(memory * BYTES_IN_MIB / 1_000_000_000),
        "g" | "gib" => Some(memory / 1024),
        // unsupported units
        _ => None,
    }
}

/// Apply replacements to the given str, where replacements are given as a list of tuples, the
/// first element as the thing to replace and the second as what to replace it with
///
/// # Arguments
///
/// * `content` - The content to apply replacements to
/// * `replacements` - The replacements to apply, a list of tuples of the str to replace and the str
///   to replace it with
#[instrument(
    name = "kvm_worker::state::overlayed::apply_xml_replacements",
    skip_all,
    err(Debug)
)]
fn apply_replacements(content: &str, replacements: &[(&str, &str)]) -> Result<String, Error> {
    // make a list of the position to replace, the placeholder, and what we're replacing it with
    let mut positions: Vec<(usize, &str, &str)> = Vec::with_capacity(replacements.len());
    // find each placeholder's position, verifying at least one occurrence
    for (placeholder, value) in replacements {
        let matches: Vec<usize> = content.match_indices(placeholder).map(|(i, _)| i).collect();
        if matches.is_empty() {
            return Err(Error::new(format!(
                "XML is missing placeholder '{placeholder}'"
            )));
        }
        for pos in matches {
            positions.push((pos, placeholder, value));
        }
    }
    // sort by position so we can walk through the content once
    positions.sort_by_key(|&(pos, _, _)| pos);
    // calculate how many characters we've added/removed
    let added: usize = positions.iter().map(|(_, _, v)| v.len()).sum();
    let removed: usize = positions.iter().map(|(_, p, _)| p.len()).sum();
    // allocate a string with the calculated final size after replacement
    let final_size = content.len() + added - removed;
    let mut result = String::with_capacity(final_size);
    // iterate through the original, replacing placeholders with their values
    let mut cursor = 0;
    for (pos, placeholder, value) in &positions {
        result.push_str(&content[cursor..*pos]);
        result.push_str(value);
        cursor = pos + placeholder.len();
    }
    result.push_str(&content[cursor..]);

    Ok(result)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // XML replacement tests

    // Sample VM XML template used across multiple tests
    const VM_TEMPLATE: &str = r"<domain type='kvm'>
  <name>{NAME}</name>
  <memory>{MEMORY}</memory>
  <vcpu placement='static'>{CPU}</vcpu>
  <devices>
    <disk type='file' device='disk'>
      <source file='{FILE}'/>
    </disk>
  </devices>
</domain>";

    #[test]
    fn detects_mem_units_default() {
        // set memory to 1024 MiB
        let memory = 1024;
        // make sure we return None since there were no units specified
        let mem_units = detect_mem_units(VM_TEMPLATE, memory);
        assert_eq!(mem_units, None);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn detects_mem_units() {
        const BYTES_IN_MIB: u64 = 1 << 20;
        // set memory to 1024 MiB
        let memory = 1024;
        // see if we detect all units correctly
        let mem_b = detect_mem_units("<memory unit='b'>{MEMORY}</memory>", memory);
        assert_eq!(mem_b, Some(memory * BYTES_IN_MIB));
        let mem_kb = detect_mem_units("<memory unit='KB'>{MEMORY}</memory>", memory);
        assert_eq!(mem_kb, Some(memory * BYTES_IN_MIB / 1000));
        let mem_kib = detect_mem_units("<memory unit='KiB'>{MEMORY}</memory>", memory);
        assert_eq!(mem_kib, Some(memory * 1024));
        let mem_mb = detect_mem_units("<memory unit='mb'>{MEMORY}</memory>", memory);
        assert_eq!(mem_mb, Some(memory * BYTES_IN_MIB / 1_000_000));
        let mem_mib = detect_mem_units("<memory unit='m'>{MEMORY}</memory>", memory);
        assert_eq!(mem_mib, Some(memory));
        let mem_gb = detect_mem_units("<memory unit='gb'>{MEMORY}</memory>", memory);
        assert_eq!(mem_gb, Some(memory * BYTES_IN_MIB / 1_000_000_000));
        let mem_gib = detect_mem_units("<memory unit='gib'>{MEMORY}</memory>", memory);
        assert_eq!(mem_gib, Some(memory / 1024));
        let mem_bad = detect_mem_units("<memory unit='BAD'>{MEMORY}</memory>", memory);
        assert_eq!(mem_bad, None);
        // check with other elements
        let mem_mb = detect_mem_units(
            "<example><memory unit='mib'>{MEMORY}</memory></example>",
            memory,
        );
        assert_eq!(mem_mb, Some(memory));
        // check with double quotes
        let mem_mb = detect_mem_units(r#"<memory unit="mib">{MEMORY}</memory>"#, memory);
        assert_eq!(mem_mb, Some(memory));
    }

    #[test]
    fn replaces_single_vm_placeholder() {
        let content = "<name>{NAME}</name>";
        let result = apply_replacements(content, &[("{NAME}", "ubuntu-vm")]).unwrap();
        assert_eq!(result, "<name>ubuntu-vm</name>");
    }

    #[test]
    fn replaces_all_vm_placeholders() {
        let replacements = [
            ("{NAME}", "ubuntu-server"),
            ("{MEMORY}", "4194304"),
            ("{CPU}", "4"),
            ("{FILE}", "/var/lib/libvirt/images/ubuntu-server.qcow2"),
        ];
        let result = apply_replacements(VM_TEMPLATE, &replacements).unwrap();

        assert!(result.contains("<name>ubuntu-server</name>"));
        assert!(result.contains("<memory>4194304</memory>"));
        assert!(result.contains("<vcpu placement='static'>4</vcpu>"));
        assert!(result.contains("source file='/var/lib/libvirt/images/ubuntu-server.qcow2'"));
        // Make sure no placeholders are left behind
        assert!(!result.contains('{'));
    }

    #[test]
    fn replacement_shorter_than_placeholder() {
        // {CUR_MEM} is 9 chars, "1024" is 4 — tests the underflow case
        let content = "<currentMemory>{CUR_MEM}</currentMemory><vcpu>{CPU}</vcpu>";
        let replacements = [("{CUR_MEM}", "1024"), ("{CPU}", "2")];
        let result = apply_replacements(content, &replacements).unwrap();
        assert_eq!(result, "<currentMemory>1024</currentMemory><vcpu>2</vcpu>");
    }

    #[test]
    fn replacement_longer_than_placeholder() {
        // {FILE} is 6 chars, the path is much longer
        let content = "<source file='{FILE}'/>";
        let replacements = [(
            "{FILE}",
            "/var/lib/libvirt/images/very-long-vm-name-here.qcow2",
        )];
        let result = apply_replacements(content, &replacements).unwrap();
        assert_eq!(
            result,
            "<source file='/var/lib/libvirt/images/very-long-vm-name-here.qcow2'/>"
        );
    }

    #[test]
    fn placeholders_in_reverse_order_in_content() {
        // Replacements declared NAME then FILE, but FILE appears first in content
        let content = "<source file='{FILE}'/><name>{NAME}</name>";
        let replacements = [("{NAME}", "test-vm"), ("{FILE}", "/images/test.qcow2")];
        let result = apply_replacements(content, &replacements).unwrap();
        assert_eq!(
            result,
            "<source file='/images/test.qcow2'/><name>test-vm</name>"
        );
    }

    #[test]
    fn errors_when_placeholder_not_found() {
        // Template has {NAME} but we look for {HOSTNAME}
        let content = "<name>{NAME}</name>";
        let result = apply_replacements(content, &[("{HOSTNAME}", "vm-01")]);
        match result {
            Err(Error::Generic(err)) => assert!(err.contains("missing")),
            other => panic!("expected Generic error, got {other:?}"),
        }
    }

    #[test]
    fn replaces_placeholder_appearing_multiple_times() {
        // {MEMORY} used for both memory and currentMemory
        let content = "<memory>{MEMORY}</memory><currentMemory>{MEMORY}</currentMemory>";
        let result = apply_replacements(content, &[("{MEMORY}", "8388608")]).unwrap();
        assert_eq!(
            result,
            "<memory>8388608</memory><currentMemory>8388608</currentMemory>"
        );
    }

    #[test]
    fn replaces_mixed_single_and_multiple_occurrences() {
        let content =
            "<name>{NAME}</name><memory>{MEMORY}</memory><currentMemory>{MEMORY}</currentMemory>";
        let replacements = [("{NAME}", "vm-01"), ("{MEMORY}", "4194304")];
        let result = apply_replacements(content, &replacements).unwrap();
        assert_eq!(
            result,
            "<name>vm-01</name><memory>4194304</memory><currentMemory>4194304</currentMemory>"
        );
    }

    #[test]
    fn errors_on_first_missing_placeholder() {
        // {NAME} exists, {DISK} does not
        let content = "<name>{NAME}</name>";
        let replacements = [("{NAME}", "vm-01"), ("{DISK}", "/images/vm.qcow2")];
        let result = apply_replacements(content, &replacements);
        match result {
            Err(Error::Generic(err)) => assert!(err.contains("missing")),
            other => panic!("expected Generic error, got {other:?}"),
        }
    }

    #[test]
    fn empty_replacements_returns_unchanged() {
        let content = "<domain><name>static-vm</name></domain>";
        let result = apply_replacements(content, &[]).unwrap();
        assert_eq!(result, content);
    }

    #[test]
    fn adjacent_placeholders() {
        // Edge case: placeholders directly next to each other
        let content = "{NAME}{CPU}";
        let replacements = [("{NAME}", "vm"), ("{CPU}", "4")];
        let result = apply_replacements(content, &replacements).unwrap();
        assert_eq!(result, "vm4");
    }

    #[test]
    fn realistic_full_vm_definition() {
        // End-to-end test with realistic libvirt values
        let replacements = [
            ("{NAME}", "web-server-01"),
            ("{MEMORY}", "8388608"), // 8 GiB in KiB
            ("{CPU}", "8"),
            ("{FILE}", "/var/lib/libvirt/images/web-server-01.qcow2"),
        ];
        let result = apply_replacements(VM_TEMPLATE, &replacements).unwrap();

        // Spot-check a few key values landed in the right places
        assert!(result.contains("<name>web-server-01</name>"));
        assert!(result.contains("<vcpu placement='static'>8</vcpu>"));
        assert!(!result.contains('{'), "no placeholders should remain");
        assert!(!result.contains('}'), "no placeholders should remain");
    }

    #[test]
    fn replaces_with_units() {
        // 4096 MiB
        let memory = 4096;
        let mem_units = detect_mem_units(VM_TEMPLATE, memory).unwrap_or(memory * 1024);
        let mem_str = mem_units.to_string();
        // End-to-end test with realistic libvirt values
        let replacements = [
            ("{NAME}", "web-server-01"),
            ("{MEMORY}", mem_str.as_str()),
            ("{CPU}", "8"),
            ("{FILE}", "/var/lib/libvirt/images/web-server-01.qcow2"),
        ];
        let result = apply_replacements(VM_TEMPLATE, &replacements).unwrap();
        // Spot-check a few key values landed in the right places
        assert!(result.contains("<name>web-server-01</name>"));
        assert!(result.contains(&format!("<memory>{mem_str}</memory>")));
        assert!(result.contains("<vcpu placement='static'>8</vcpu>"));
        assert!(!result.contains('{'), "no placeholders should remain");
        assert!(!result.contains('}'), "no placeholders should remain");
    }
}
