use args::Args;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thorium::models::{Reaction, ReactionRequest};
use thorium::{Thorium, models::GenericJobArgs};
use uuid::Uuid;
use yara_x::{Rule, Rules};

mod args;

pub struct VolFan {
    /// A Thorium client for spawning sub reactions
    thorium: Thorium,
    /// The rules to scan with
    rules: Rules,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum OsKinds {
    // A Windows memory image
    Windows,
    /// A Linux memory image
    Linux,
    /// A mac memory image
    Mac,
}

impl OsKinds {
    /// Get the modules to run for each ``OsKind``
    pub fn modules(&self) -> Vec<&str> {
        // start with an empty list of modules
        let mut modules = Vec::with_capacity(50);
        // add the right modules based on our os kind
        match self {
            OsKinds::Windows => {
                // add some windows modules to spawn
                modules.push("windows.info");
                modules.push("windows.pslist");
                modules.push("windows.pstree");
                modules.push("windows.sessions");
                modules.push("windows.modules");
                modules.push("windows.dlllist");
                modules.push("windows.filescan");
                modules.push("windows.netscan");
                modules.push("windows.cmdline");
                modules.push("windows.sockets");
                modules.push("windows.malfind");
                modules.push("windows.getsids");
                modules.push("windows.scheduled_tasks");
                modules.push("windows.registry.hivelist");
                modules.push("windows.registry.printkey");
                modules.push("windows.handles");
                modules.push("windows.svcscan");
                modules.push("windows.netstat");
                modules.push("windows.privileges");
                modules.push("windows.vaddump");
                modules.push("windows.envars.Envars");
                modules.push("windows.scheduled_tasks.ScheduledTasks");
                modules.push("windows.truecrypt.Passphrase");
                modules.push("windows.malware.direct_system_calls.DirectSystemCalls");
                modules.push("windows.malware.drivermodule.DriverModule");
                modules.push("windows.malware.hollowprocesses.HollowProcesses");
                modules.push("windows.malware.indirect_system_calls.IndirectSystemCalls");
                modules.push("windows.malware.ldrmodules.LdrModules");
                modules.push("windows.malware.malfind.Malfind");
                modules.push("windows.malware.processghosting.ProcessGhosting");
                modules.push("windows.malware.psxview.PsXView");
                modules.push("windows.malware.skeleton_key_check.Skeleton_Key_Check");
                modules.push("windows.malware.suspicious_threads.SuspiciousThreads");
                modules.push("windows.malware.svcdiff.SvcDiff");
                modules.push("windows.malware.unhooked_system_calls.UnhookedSystemCalls");
            }
            OsKinds::Linux => {
                // add some linux modules to spawn
                modules.push("linux.bash.Bash");
                modules.push("linux.boottime.Boottime");
                modules.push("linux.capabilities.Capabilities");
                modules.push("linux.check_afinfo.Check_afinfo");
                modules.push("linux.check_creds.Check_creds");
                modules.push("linux.check_idt.Check_idt");
                modules.push("linux.check_modules.Check_modules");
                modules.push("linux.check_syscall.Check_syscall");
                modules.push("linux.ebpf.EBPF");
                modules.push("linux.elfs.Elfs");
                modules.push("linux.envars.Envars");
                modules.push("linux.graphics.fbdev.Fbdev");
                modules.push("linux.hidden_modules.Hidden_modules");
                modules.push("linux.ip.Addr");
                modules.push("linux.ip.Link");
                modules.push("linux.kallsyms.Kallsyms");
                modules.push("linux.keyboard_notifiers.Keyboard_notifiers");
                modules.push("linux.kmsg.Kmsg");
                modules.push("linux.kthreads.Kthreads");
                modules.push("linux.library_list.LibraryList");
                modules.push("linux.lsmod.Lsmod");
                modules.push("linux.lsof.Lsof");
                modules.push("linux.malfind.Malfind");
                modules.push("linux.malware.check_afinfo.Check_afinfo");
                modules.push("linux.malware.check_creds.Check_creds");
                modules.push("linux.malware.check_idt.Check_idt");
                modules.push("linux.malware.check_modules.Check_modules");
                modules.push("linux.malware.check_syscall.Check_syscall");
                modules.push("linux.malware.hidden_modules.Hidden_modules");
                modules.push("linux.malware.keyboard_notifiers.Keyboard_notifiers");
                modules.push("linux.malware.malfind.Malfind");
                modules.push("linux.malware.modxview.Modxview");
                modules.push("linux.malware.netfilter.Netfilter");
                modules.push("linux.malware.tty_check.Tty_Check");
                modules.push("linux.module_extract.ModuleExtract");
                modules.push("linux.modxview.Modxview");
                modules.push("linux.mountinfo.MountInfo");
                modules.push("linux.netfilter.Netfilter");
                modules.push("linux.pagecache.Files");
                modules.push("linux.pagecache.InodePages");
                modules.push("linux.pagecache.RecoverFs");
                modules.push("linux.pidhashtable.PIDHashTable");
                modules.push("linux.proc.Maps");
                modules.push("linux.psaux.PsAux");
                modules.push("linux.pscallstack.PsCallStack");
                modules.push("linux.pslist.PsList");
                modules.push("linux.psscan.PsScan");
                modules.push("linux.pstree.PsTree");
                modules.push("linux.ptrace.Ptrace");
                modules.push("linux.sockstat.Sockstat");
                modules.push("linux.tracing.ftrace.CheckFtrace");
                modules.push("linux.tracing.perf_events.PerfEvents");
                modules.push("linux.tracing.tracepoints.CheckTracepoints");
                modules.push("linux.tty_check.tty_check");
                modules.push("linux.vmcoreinfo.VMCoreInfo");
            }
            OsKinds::Mac => {
                // add some mac modules to spawn
                modules.push("mac.bash.Bash");
                modules.push("mac.check_syscall.Check_syscall");
                modules.push("mac.check_sysctl.Check_sysctl");
                modules.push("mac.check_trap_table.Check_trap_table");
                modules.push("mac.dmesg.Dmesg");
                modules.push("mac.ifconfig.Ifconfig");
                modules.push("mac.kauth_listeners.Kauth_listeners");
                modules.push("mac.kauth_scopes.Kauth_scopes");
                modules.push("mac.kevents.Kevents");
                modules.push("mac.list_files.List_Files");
                modules.push("mac.lsmod.Lsmod");
                modules.push("mac.lsof.Lsof");
                modules.push("mac.malfind.Malfind");
                modules.push("mac.mount.Mount");
                modules.push("mac.netstat.Netstat");
                modules.push("mac.proc_maps.Maps");
                modules.push("mac.psaux.Psaux");
                modules.push("mac.pslist.PsList");
                modules.push("mac.pstree.PsTree");
                modules.push("mac.socket_filters.Socket_filters");
                modules.push("mac.timers.Timers");
                modules.push("mac.trustedbsd.Trustedbsd");
                modules.push("mac.vfsevents.VFSevents");
            }
        }
        modules
    }
}

impl From<Rule<'_, '_>> for OsKinds {
    /// Convert a rule hit to an ``OsKind``
    fn from(rule: Rule) -> Self {
        match rule.identifier() {
            "Windows_Memory_Image" => OsKinds::Windows,
            "Linux_Memory_Image" => OsKinds::Linux,
            "Mac_Memory_Image" => OsKinds::Mac,
            unknown => panic!("Unknown Rule: {unknown}"),
        }
    }
}

impl VolFan {
    /// Create a new Volaltility Fan Outer
    async fn new(args: &Args) -> Self {
        // create a thorium client from either a ctl conf or keys
        let thorium = match &args.keys {
            Some(keys) => Thorium::from_key_file(keys)
                .await
                .expect("Failed to create Thorium client from keys"),
            None => Thorium::from_ctl_conf_file(&args.conf)
                .await
                .expect("Failed to create Thorium client from ctlconf"),
        };
        // compile our yara rules from disk
        let mut compiler = yara_x::Compiler::new();
        // go through and read in all of our rule files
        let win_rule =
            std::fs::read_to_string("rules/windows.yar").expect("Failed to load windows rules");
        compiler.add_source(win_rule.as_str()).unwrap();
        let linux_rule =
            std::fs::read_to_string("rules/linux.yar").expect("Failed to load linux rules");
        compiler.add_source(linux_rule.as_str()).unwrap();
        // compile our rules
        let rules = compiler.build();
        // build our volatility fanner
        VolFan { thorium, rules }
    }

    /// Get our reaction info
    pub async fn get_reaction_info(&self, reaction: Uuid, group: &str) -> Reaction {
        // get our reaction info
        self.thorium
            .reactions
            .get(&group, reaction)
            .await
            .expect("Failed to get reaction info")
    }

    /// Probe a memory image for the correct os
    fn probe(&self, target: &PathBuf) -> Vec<OsKinds> {
        println!("Probing {}", target.display());
        // build a yara scanner with our rules
        let mut scanner = yara_x::Scanner::new(&self.rules);
        // scan our target memory image and try to determine what OS its from
        let scan_res = scanner
            .scan_file(target)
            .unwrap_or_else(|_| panic!("Failed to scan {}", target.display()));
        // convert our rule hits into the os kinds
        scan_res.matching_rules().map(OsKinds::from).collect()
    }

    /// Write our detected os kinds to disk
    pub async fn write_os_kinds(&self, os_kinds: &Vec<OsKinds>) {
        // serialize our list of os kinds
        let serialize = serde_json::to_string(os_kinds).expect("Failed to serialize os kind list");
        // instance a map with space for our os kinds
        let mut map = HashMap::with_capacity(1);
        // add our ous kinds to our generic cache map
        map.insert("OsKinds", serialize);
        // serialize our map
        let serialized_map =
            serde_json::to_string(&map).expect("Failed to serialize generic cache");
        // write our serialized map to our cache on disk
        tokio::fs::write("/tmp/thorium/cache/generic.json", serialized_map.as_bytes())
            .await
            .expect("Failed to write generic cache to disk");
    }

    /// Analyze and fan out jobs for this memory image
    ///
    /// # Panics
    ///
    /// Panics if we can't create reactions.
    pub async fn analyze(&self, target: &PathBuf, reaction: Reaction, job: Uuid) {
        // get our target sha2x56
        let sha256 = reaction.samples.first().expect("Reaction has no samples?");
        // pre allocate a vec for our bulk spawned reactions
        let mut reqs = Vec::with_capacity(50);
        // build a base reaction request for this module
        let reaction_req = ReactionRequest::new(reaction.group.as_str(), "auto-volatility3-worker")
            .sample(sha256)
            .tag(sha256)
            .parent(reaction.id);
        // probe this memory image to determine the os kind
        let os_kinds = self.probe(target);
        // spawn jobs for all of our detected os kind
        for os_kind in &os_kinds {
            println!("Detected: {os_kind:?}");
            // get the modules to run for each os kind
            for module in os_kind.modules() {
                println!("Building reaction request for {module}");
                // build the file name to write our results too
                let output = format!("{}.txt", module.replace('.', "_"));
                // set the module for this sub reaction
                let specific = reaction_req.clone().args(
                    "auto-volatility3-worker",
                    GenericJobArgs::default().positionals(vec![module, &output]),
                );
                // add this specific request to our list of reactions to create
                reqs.push(specific);
            }

        }
        println!("Creating {} reactions", reqs.len());
        // create sub reactions for all of the requested modules
        let create_resp = self
            .thorium
            .reactions
            .create_bulk(&reqs)
            .await
            .expect("Failed to create sub reactions");
        // log our sub reactions
        for id in create_resp.created {
            println!("Created subreaction: {id}");
        }
        // log any errors
        for (index, msg) in &create_resp.errors {
            println!("Error creating reaction {index}: {msg}");
        }
        // if we have any failed reaction creates then fail this generator
        if !create_resp.errors.is_empty() {
            // abort this generator with a helpful error message
            panic!(
                "Failed to create {} out of {} reactions",
                create_resp.errors.len(),
                reqs.len()
            );
        }
        // save our os kinds to our generic cache
        self.write_os_kinds(&os_kinds).await;
        // sleep our reaction
        self.thorium
            .jobs
            .sleep(&job, "Analyzed")
            .await
            .expect("Failed to sleep reaction");
    }

    /// Write our results to disk for Thorium to pickup
    pub async fn submit(&self) {
        // load our cache from disk
        let cache_str = tokio::fs::read_to_string("/tmp/thorium/cache/generic.json")
            .await
            .expect("Failed to load generic cache");
        // deserialize our cache
        let cache: HashMap<String, String> =
            serde_json::from_str(&cache_str).expect("Failed to deseralize generic cache");
        // deserialize our os kinds
        let os_kinds: Vec<OsKinds> =
            serde_json::from_str(&cache["OsKinds"]).expect("Failed to deserialize os kinds");
        // write our windows info results to our results file if we have windows results
        if os_kinds.contains(&OsKinds::Windows) {
            // rename our windows info results to our results file
            tokio::fs::rename(
                "/tmp/thorium/cache/files/windows_info.txt",
                "/tmp/thorium/results",
            )
            .await
            .expect("Failed to copy windows_info.txt to results");
        }
        // the path to store our result files at
        let result_file_path = PathBuf::from("/tmp/thorium/result-files");
        // get a stream of all files in this directory
        let mut entries = tokio::fs::read_dir("/tmp/thorium/cache/files")
            .await
            .expect("Failed to get stream of cache file entries");
        // step over the items in this stream
        while let Some(entry) = entries.next_entry().await.unwrap() {
            // get this entries path
            let path = entry.path();
            // skip anything thats not a file
            if path.is_file() {
                // build the new path to write this file too
                let new = result_file_path.join(path.file_name().unwrap());
                // move this result file from our cache to result files
                tokio::fs::rename(path, new)
                    .await
                    .expect("Failed to write result file");
            }
        }
    }

    /// Analyze or submit results for a specific reaction
    pub async fn process(&self, target: &PathBuf, reaction: Uuid, job: Uuid, group: &str) {
        // get our reaction info
        let reaction = self.get_reaction_info(reaction, group).await;
        // if we have cache info then we have already ran
        if reaction.has_cache {
            self.submit().await;
        } else {
            // determine the right os type for thismemory image and then start up analysis jobs
            self.analyze(target, reaction, job).await;
        }
    }
}

#[tokio::main]
async fn main() {
    // get the args to use
    let args = Args::parse();
    // build a volatility fan outer
    let volfan = VolFan::new(&args).await;
    // process this memory image
    volfan.process(&args.target, args.reaction, args.job, &args.group).await;
}
