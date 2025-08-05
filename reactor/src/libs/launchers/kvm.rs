//! Launches KVM vms for Thorium

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use thorium::models::{Image, Node, Worker};
use thorium::{Error, Thorium};
use tokio::process::Command;
use tracing::{span, Level, Span};
use virt::connect::Connect;
use virt::domain::Domain;
use virt::domain_snapshot::DomainSnapshot;

use super::Launcher;
use crate::libs::keys;

pub struct Kvm {
    /// The kvm specific args
    args: crate::args::Kvm,
    /// The inactive domains in a golden state we can spawn new workers on
    golden: HashMap<String, HashMap<String, BTreeSet<String>>>,
}

impl Kvm {
    /// Create a new kvm connector
    ///
    /// # Arguments
    ///
    /// * `args` - The args for the kvm launcher
    pub fn new(args: &crate::args::Kvm) -> Result<Kvm, Error> {
        // build our kvm launcher
        let kvm = Kvm {
            args: args.clone(),
            golden: HashMap::default(),
        };
        Ok(kvm)
    }

    /// Find a golden vm for this image type
    ///
    /// # Arguments
    ///
    /// * `image` - The image to find a golden image slot for
    fn find_golden(&mut self, image: &Image) -> Option<String> {
        // get this images group set
        if let Some(image_set) = self.golden.get_mut(&image.group) {
            // get the golden vms set
            if let Some(golden_set) = image_set.get_mut(&image.name) {
                // pop the first golden vm name
                return golden_set.pop_first();
            }
        }
        // we did not find a golden vm
        None
    }

    /// Build the xml to attach an iso
    ///
    /// # Arguments
    ///
    /// * `worker` - The worker to attach an iso to
    /// * `span` - The span to log traces under
    async fn build_iso_xml(&self, worker: &Worker, span: &Span) -> Result<String, Error> {
        // start our build iso span
        span!(parent: span, Level::INFO, "Build And Attach ISO");
        // build the path to this users keys
        let mut keys = keys::path(&worker.user);
        // get the folder for our keys
        keys.pop();
        // build the path to the agent
        keys.push("thorium.exe");
        // copy the agent to this folder as well
        tokio::fs::copy("/opt/thorium-windows/thorium-agent.exe", &keys).await?;
        // go back to the keys folder
        keys.pop();
        // build the path to write our new iso file too
        let mut tmp_iso = PathBuf::new();
        tmp_iso.push(&self.args.temp);
        tmp_iso.push(&worker.name);
        tmp_iso.set_extension("iso");
        println!(
            "mkisofs -o {} {}",
            &tmp_iso.to_string_lossy(),
            &keys.to_string_lossy()
        );
        // build our make iso command
        Command::new("mkisofs")
            .arg("-o")
            .arg(&tmp_iso)
            .arg(&keys)
            .spawn()?
            .wait()
            .await?;
        // build the xml to attach this iso
        let xml = format!(
            "<disk type=\"file\" device=\"cdrom\"> \
                <driver name=\"qemu\" type=\"raw\"/> \
                <source file=\"{iso}\"/> \
                <target dev=\"sdb\" bus=\"sata\"/> \
                <readonly/> \
                <address type=\"drive\" controller=\"0\" bus=\"0\" target=\"0\" unit=\"1\"/> \
            </disk>",
            iso = tmp_iso.to_string_lossy()
        );
        Ok(xml)
    }

    /// Convert a golden vm into an active worker VM
    ///
    /// # Arguments
    ///
    /// * `golden` - The name of the golden image that was used
    /// * `worker` - The worker we are assigning a golden image too
    /// * `span` - The span to log traces under
    pub async fn convert(
        &mut self,
        golden: String,
        worker: &Worker,
        span: &Span,
    ) -> Result<(), Error> {
        // start our start kvm domain span
        let span = span!(parent: span, Level::INFO, "Convert Golden Domain", name = &worker.name);
        // build the iso for this vm
        let xml = self.build_iso_xml(worker, &span).await?;
        // connect to our kvm daemon
        let client = Connect::open(&self.args.socket).unwrap();
        // get the domain for our vm
        let domain = Domain::lookup_by_name(&client, &golden).unwrap();
        // rename our vm
        domain.rename(&worker.name, 0).unwrap();
        // get the golden snapshot for this vm
        let snapshot = DomainSnapshot::lookup_by_name(&domain, "golden", 0).unwrap();
        // revert to our snapshot
        snapshot.revert(0).unwrap();
        // attach our iso
        domain.attach_device(&xml).unwrap();
        Ok(())
    }

    /// Get the vncdisplay for a target vm
    ///
    /// # Arguments
    ///
    /// * `domain` - The domain to get a vnc display address for
    pub async fn get_vnc_display(&self, domain: &str) -> Result<String, Error> {
        // build and execute the command to get this domains vnc display port
        let output = Command::new("virsh")
            .args(&["vncdisplay", domain])
            .output()
            .await?;
        // throw an error if we failed to get our vnc display port
        if output.status.success() {
            // get the stdout for this command as a string
            let untrimmed = String::from_utf8_lossy(&output.stdout);
            // trim any newlines
            Ok(untrimmed.trim().to_owned())
        } else {
            Err(Error::new(format!(
                "Failed to get vnc display port for {}",
                domain
            )))
        }
    }

    /// Launch the agent on the target vm
    ///
    /// # Arguments
    ///
    /// * `domain` - The display to launch an agent over
    /// * `worker` - The worker we are launching an agent for
    async fn launch_agent(&self, display: &str, worker: &Worker) -> Result<(), Error> {
        // build the cmd to spawn our agent
        let cmd = format!(
            "\\THORIUM.EXE --cluster {cluster} --node {node} \
            --group {group} --pipeline {pipeline} --stage {stage} --name {name} \
            --keys D",
            cluster = &worker.cluster,
            node = &worker.node,
            group = &worker.group,
            pipeline = &worker.pipeline,
            stage = &worker.stage,
            name = &worker.name,
        );
        println!("args -> D:{}:\\KEYS.YML kvm", cmd);
        // execute the command to launch our agent
        let output = Command::new("vncdotool")
            .args(&[
                "-s",
                display,
                "type",
                "D",
                "key",
                "shift-:",
                "type",
                &cmd,
                "key",
                "shift-:",
                "type",
                "\\KEYS.YML kvm",
                "key",
                "enter",
            ])
            .output()
            .await?;
        // throw an error if we failed to spawn our agent
        if output.status.success() {
            Ok(())
        } else {
            Err(Error::new(format!(
                "Failed to send vncdotool command to {}",
                display
            )))
        }
    }
}

/// Shutdown and rename a vm
///
/// # Arguments
///
/// * `name` -
fn shutdown_vm(name: &str, args: &crate::args::Kvm, span: &Span) -> Result<(), Error> {
    // start our shutdown vm span
    span!(parent: span, Level::INFO, "Shutdown VM", vm=name);
    // build the path to delete our iso file at
    let mut tmp_iso = PathBuf::new();
    tmp_iso.push(&args.temp);
    tmp_iso.push(name);
    tmp_iso.set_extension("iso");
    // remove the old iso file if it exists
    if std::fs::metadata(&tmp_iso).is_ok() {
        // remove the old iso file
        std::fs::remove_file(&tmp_iso)?;
    }
    // connect to our kvm daemon
    let client = Connect::open(&args.socket).unwrap();
    // get the domain for our vm
    if let Ok(domain) = Domain::lookup_by_name(&client, name) {
        // only shutdown running vms
        if domain.is_active().unwrap() {
            // shutdown our vm
            domain.shutdown().unwrap();
            // every 10 seconds send the shutdown command again
            let mut checks = 0;
            // wait until this vm is shutdown
            while domain.is_active().unwrap() {
                // sleep for 1 second
                std::thread::sleep(std::time::Duration::from_secs(1));
                // increment our check count
                checks += 1;
                // if we have checked 10 times then resend our shutdown command
                if checks >= 10 {
                    // send our shutdown command
                    domain.shutdown().unwrap();
                    // reset our check count
                    checks = 0;
                }
            }
            // build our new vm name
            let new_name = format!("thorium_corn_test");
            // rename our vm
            domain.rename(&new_name, 0).unwrap();
        }
    }
    Ok(())
}

#[async_trait::async_trait]
impl Launcher for Kvm {
    /// Spawn a worker and then return a process id that can be used to track it
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `worker` - The worker to launch
    /// * `span` - The span to log traces under
    async fn launch(
        &mut self,
        thorium: &Thorium,
        worker: &Worker,
        span: &Span,
    ) -> Result<(), Error> {
        // start our launch kvm job span
        let span = span!(parent: span, Level::INFO, "Launch Kvm Worker");
        // get the image info for this worker
        let image = thorium.images.get(&worker.group, &worker.stage).await?;
        // try to find a golden vm for this job
        if let Some(golden) = self.find_golden(&image) {
            // try to define and spawn a new vm
            self.convert(golden, &worker, &span).await?;
            // get the vnc display for this vm
            let vnc_display = self.get_vnc_display(&worker.name).await?;
            println!("vnc port -> {}", vnc_display);
            // launch our agent
            self.launch_agent(&vnc_display, worker).await?;
        }
        Ok(())
    }

    /// Check if any of our current workers have completed or died
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `info` - Info about our node and its workers
    /// * `active` - The names of the currently active workers in the reactor
    /// * `span` - The span to log traces under
    async fn check(
        &mut self,
        _thorium: &Thorium,
        info: &mut Node,
        active: &mut HashMap<String, Worker>,
        span: &Span,
    ) -> Result<(), Error> {
        // start our launch kvm job span
        let span = span!(parent: span, Level::INFO, "Check KVM Workers");
        // connect to our kvm daemon
        let client = Connect::open(&self.args.socket).unwrap();
        // build a map of existing vms
        let mut existing = HashSet::with_capacity(50);
        // build the flags for listing domains
        let flags = virt::sys::VIR_CONNECT_LIST_DOMAINS_ACTIVE
            | virt::sys::VIR_CONNECT_LIST_DOMAINS_INACTIVE;
        // clear our golden images
        self.golden.clear();
        // remove any no longer active workers
        active.retain(|name, _| info.workers.contains_key(name));
        // crawl over the domains on this node
        for dom in client.list_all_domains(flags).unwrap() {
            let name = dom.get_name().unwrap_or_else(|_| String::from("no-name"));
            // if this worker starts with the name thorium-golden then add it to our golden list
            if name.starts_with("thorium_") {
                println!("domain -> {}", name);
                // split our string by - to extract the group and image name
                let mut split = name.split('_').skip(1);
                // get our group name
                if let Some(group) = split.next() {
                    // get our image name
                    if let Some(image) = split.next() {
                        // get an entry to this groups images
                        let image_entry = self.golden.entry(group.to_owned()).or_default();
                        // get an entry to our golden vms
                        let golden_entry = image_entry.entry(image.to_owned()).or_default();
                        // insert our golden image
                        golden_entry.insert(name.clone());
                    }
                }
            }
            existing.insert(name);
        }
        println!("active -> {:#?}", active);
        // shutdown any vms that are no longer tied to workers
        for name in existing.iter() {
            println!("maybe shutdown? -> {}", name);
            println!(
                "{} && {} && {}",
                !active.contains_key(name),
                name != "thorium_corn_test",
                name != "win10-james"
            );
            // shutdown any vms not in our existing set
            if !active.contains_key(name) && name != "thorium_corn_test" && name != "win10-james" {
                println!("shutdown -> {}", name);
                // shutdown our vm
                shutdown_vm(&name, &self.args, &span)?;
            }
        }
        // drop any workers that do not still exist
        active.retain(|name, _| existing.contains(name));
        Ok(())
    }

    /// Shutdown a list of workers
    ///
    /// # Arguments
    ///
    /// * `thorium` - A Thorium client
    /// * `workers` - The workers to shutdown
    /// * `span` - The span to log traces under
    async fn shutdown(
        &mut self,
        _thorium: &Thorium,
        mut _workers: HashSet<String>,
        _span: &Span,
    ) -> Result<(), Error> {
        panic!("AHHHHH");
    }
}
