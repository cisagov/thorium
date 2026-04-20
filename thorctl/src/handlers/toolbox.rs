//! Handles toolbox commands

use thorium::Error;

mod build;
mod build_images;
mod collisions;
mod diff;
mod export;
mod import;
pub(crate) mod init;
mod manifest;
pub(crate) mod policies;
mod prompt;
mod remove;
mod shared;

use crate::args::Args;
use crate::args::toolbox::Toolbox;
use crate::utils;

/// Dispatches a toolbox subcommand to its handler
///
/// # Arguments
///
/// * `args` - The top-level thorctl args
/// * `toolbox` - The toolbox subcommand to execute
pub async fn handle(args: &Args, toolbox: &Toolbox) -> Result<(), Error> {
    // Dispatch the three client-free subcommands first so we never build an API
    // client (or run the insecure/update checks) for work that is purely local.
    if let Toolbox::Build(cmd) = toolbox {
        // build walks the tree with synchronous std::fs, so run it off the async
        // runtime on a blocking thread (mirrors the spawn_blocking wrap in export
        // and diff); clone the command because the spawned task must own its input.
        let cmd = cmd.clone();
        // A JoinError here means the blocking task itself panicked, which we surface
        // as an error rather than letting the panic propagate out of the runtime.
        return tokio::task::spawn_blocking(move || build::build(&cmd))
            .await
            .map_err(|err| Error::new(format!("Toolbox build task panicked: {err}")))?;
    }
    if let Toolbox::Init(cmd) = toolbox {
        // init only scaffolds files on disk, so it needs no client and returns early.
        return init::handle(cmd, args).await;
    }
    if let Toolbox::BuildImages(cmd) = toolbox {
        // build-images needs a docker/podman runtime but no API client, so load the
        // ctl config best-effort purely to read `container_runtime`; if the config is
        // missing or unreadable, fall back to the flag then PATH auto-detection.
        let configured_runtime = thorium::CtlConf::from_path(&args.config)
            .ok()
            .and_then(|conf| conf.container_runtime);
        // Resolve and cache the runtime (flag -> config -> PATH) before any build runs.
        super::container::init_runtime(args.container_runtime, configured_runtime);
        return build_images::build_images(cmd).await;
    }
    // Everything below requires an API client, so build it (this also loads the
    // full ctl config we reuse for the runtime and the warning/update gates).
    let (conf, thorium) = utils::get_client(args).await?;
    // Warn about an insecure (non-TLS / untrusted) connection unless the config
    // explicitly opts out of the warning.
    if !conf.skip_insecure_warning.unwrap_or_default() {
        utils::warn_insecure_conf(&conf)?;
    }
    // Offer a client/server version-update check unless either the CLI flag or the
    // config disables it; both must allow it for the prompt to run.
    if !args.skip_update && !conf.skip_update.unwrap_or_default() {
        crate::handlers::update::ask_update(&thorium).await?;
    }
    match toolbox {
        Toolbox::Import(cmd) => {
            // Resolve the runtime up front so pushing any bundled image tarballs has a
            // CLI ready; non-bundled imports simply never invoke it.
            super::container::init_runtime(args.container_runtime, conf.container_runtime);
            import::import(thorium, conf, cmd, args.workers).await
        }
        Toolbox::Export(cmd) => {
            // Resolve the runtime up front so a `--with-images` export can pull/save
            // tarballs; a plain export simply never invokes it.
            super::container::init_runtime(args.container_runtime, conf.container_runtime);
            export::export(thorium, cmd, args, &conf).await
        }
        Toolbox::Remove(cmd) => remove::remove(thorium, conf, cmd).await,
        Toolbox::Diff(cmd) => {
            // Translate "drift detected" into a `git diff --exit-code`-style non-zero
            // exit here at the dispatch boundary, rather than inside diff(), so diff()
            // returns normally and all its resources drop before the process exits.
            if diff::diff(thorium, &conf, cmd).await? {
                std::process::exit(1);
            }
            Ok(())
        }
        // The three client-free variants returned earlier, so they can never reach here.
        Toolbox::Build(_) | Toolbox::Init(_) | Toolbox::BuildImages(_) => unreachable!(),
    }
}
