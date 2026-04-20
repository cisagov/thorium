//! Shared helpers for moving container images with the local container CLI
//!
//! These wrap `pull/save/load/tag/push/build` so that the image export/import
//! flows — both the per-image `images`/`pipelines` commands and the
//! `toolbox --with-images` offline bundles — share one implementation. The CLI
//! used is docker or podman, resolved once per run by [`init_runtime`] (podman is
//! CLI-compatible with docker for these verbs, so only the binary name differs).
//! All helpers shell out to that binary and return an error at runtime if it is
//! not installed. Compression is done in-process with flate2 so no external
//! `gzip` binary is required.

use flate2::Compression;
use flate2::write::GzEncoder;
use std::path::Path;
use std::process::{Output, Stdio};
use std::sync::OnceLock;
use thorium::{ContainerRuntime, Error};
use tokio::process::Command;

use super::progress::Bar;

/// The container runtime resolved once for this run and reused by every command
static RUNTIME: OnceLock<ContainerRuntime> = OnceLock::new();

/// Resolve and cache the container runtime to use for this run
///
/// Resolves with flag > config > auto-detect precedence and stores the result;
/// later calls are no-ops, which is safe since a thorctl run executes a single
/// subcommand. Call this once at the start of any flow that moves images. PATH
/// detection only runs when neither the flag nor the config picked a runtime.
///
/// # Arguments
///
/// * `flag` - The `--container-runtime` override, if given
/// * `configured` - The `container_runtime` config value, if set
pub fn init_runtime(flag: Option<ContainerRuntime>, configured: Option<ContainerRuntime>) {
    // resolve the runtime, probing PATH only when neither flag nor config chose one
    let resolved = resolve(flag, configured, detect);
    // cache it; a second call this run is a harmless no-op
    let _ = RUNTIME.set(resolved);
}

/// Choose a runtime from the flag, the config value, and a detection fallback
///
/// Precedence is the flag, then the config value, then the result of `detect`.
/// `detect` is a closure called only when neither the flag nor the config picked a
/// runtime, which both keeps PATH probing lazy and lets tests inject a result
/// without docker or podman installed.
///
/// # Arguments
///
/// * `flag` - The `--container-runtime` override, if given
/// * `configured` - The `container_runtime` config value, if set
/// * `detect` - Produces the auto-detected runtime when neither above is set
fn resolve(
    flag: Option<ContainerRuntime>,
    configured: Option<ContainerRuntime>,
    detect: impl FnOnce() -> ContainerRuntime,
) -> ContainerRuntime {
    // the explicit flag wins, then the config value, otherwise what we detect
    flag.or(configured).unwrap_or_else(detect)
}

/// Auto-detect an installed container runtime from PATH
///
/// Prefers docker, then podman, by checking whether each responds to `--version`.
/// Falls back to docker when neither is found so the eventual command surfaces a
/// clear "not installed" error.
fn detect() -> ContainerRuntime {
    // probe docker first, then podman, by running their `--version`
    for runtime in [ContainerRuntime::Docker, ContainerRuntime::Podman] {
        let found = std::process::Command::new(runtime.binary())
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
        if found {
            return runtime;
        }
    }
    // neither runtime was found; default to docker and let the real command error
    ContainerRuntime::Docker
}

/// The container runtime resolved for this run (docker until [`init_runtime`] runs)
fn runtime() -> ContainerRuntime {
    // fall back to the default (docker) when called before init_runtime so error
    // messages still name a concrete binary instead of panicking on an unset cell
    RUNTIME.get().copied().unwrap_or_default()
}

/// Build a [`Command`] for the resolved runtime with the given arguments applied
///
/// # Arguments
///
/// * `args` - The CLI arguments to pass to the runtime binary
fn command<I, S>(args: I) -> Command
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    // target the resolved runtime binary (docker or podman) for this run
    let mut cmd = Command::new(runtime().binary());
    // apply the caller's arguments; stdio handling is left to each caller
    cmd.args(args);
    cmd
}

/// Check a finished container command, surfacing its stderr on failure
///
/// # Arguments
///
/// * `output` - The finished command output to inspect
/// * `ctx` - A human-readable description of what was attempted
fn check(output: &Output, ctx: &str) -> Result<(), Error> {
    // a successful exit needs no diagnostics
    if output.status.success() {
        return Ok(());
    }
    // decode stderr lossily so non-UTF-8 runtime output can't itself become an error
    let stderr = String::from_utf8_lossy(&output.stderr);
    // surface the runtime's own stderr (trimmed) alongside our context for a usable message
    Err(Error::new(format!("{ctx}: {}", stderr.trim())))
}

/// Pull an image from a registry (`<runtime> pull <url>`)
///
/// # Arguments
///
/// * `url` - The fully-qualified image url to pull
/// * `bar` - The progress bar to update with status
pub async fn pull(url: &str, bar: &Bar) -> Result<(), Error> {
    // reflect the current step in the shared progress bar
    bar.set_message("Pulling image");
    // run `<runtime> pull <url>` and capture its output; a spawn failure here means the
    // runtime binary is missing or unrunnable, distinct from a non-zero pull exit below
    let output = command(["pull", url])
        .output()
        .await
        .map_err(|e| Error::new(format!("Failed to run {} pull: {e}", runtime())))?;
    // turn a non-zero exit into an error carrying the runtime's stderr
    check(&output, &format!("{} pull failed for '{url}'", runtime()))
}

/// Save an image to a gzipped tarball (`<runtime> save <url>` compressed with flate2)
///
/// Creates the destination's parent directory if needed.
///
/// # Arguments
///
/// * `url` - The image url to save (must already be present locally)
/// * `dest` - The `.tar.gz` path to write
/// * `bar` - The progress bar to update with status
pub async fn save(url: &str, dest: &Path, bar: &Bar) -> Result<(), Error> {
    // reflect the current step in the shared progress bar
    bar.set_message("Saving image");
    // ensure the destination's parent exists so std::fs::File::create in the blocking
    // task doesn't fail on a missing directory
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| Error::new(format!("Failed to create '{}': {e}", parent.display())))?;
    }
    // own the inputs so they can move into the blocking closure with a 'static lifetime
    let url = url.to_string();
    let dest = dest.to_path_buf();
    // spawning the subprocess, reading its tar stream, gzip-compressing it (CPU-bound),
    // and writing the file are all blocking, so run the whole pipeline on a blocking
    // thread instead of bridging a sync compressor onto the async runtime; a JoinError
    // here means the blocking task itself panicked
    tokio::task::spawn_blocking(move || save_blocking(&url, &dest))
        .await
        .map_err(|err| Error::new(format!("Image save task panicked: {err}")))?
}

/// Synchronously run `<runtime> save <url>` and gzip its tar output into `dest`
///
/// Streams the subprocess's stdout straight through the gzip encoder with
/// [`std::io::copy`] so memory stays bounded regardless of image size. Always reaps
/// the subprocess; on any failure it is killed and the partial archive removed so it
/// can't be mistaken for a complete, importable tarball. Intended to be called from a
/// blocking context (see [`save`]).
///
/// # Arguments
///
/// * `url` - The image url to save (must already be present locally)
/// * `dest` - The `.tar.gz` path to write
fn save_blocking(url: &str, dest: &Path) -> Result<(), Error> {
    // spawn the save with a piped stdout so we can compress it as it streams; stderr is
    // left inherited so the runtime's own progress/errors reach the terminal
    let mut child = std::process::Command::new(runtime().binary())
        .args(["save", url])
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| Error::new(format!("Failed to run {} save: {e}", runtime())))?;
    // take ownership of the piped stdout handle so we can stream it; absent only if the
    // pipe wasn't set up, which shouldn't happen given the Stdio::piped above
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| Error::new(format!("{} save did not produce stdout", runtime())))?;
    // stream-compress the tar into the destination; std::io::copy bounds memory by
    // copying in fixed-size chunks rather than buffering the whole image
    let stream = (|| -> Result<(), Error> {
        let gz_file = std::fs::File::create(dest)
            .map_err(|e| Error::new(format!("Failed to create '{}': {e}", dest.display())))?;
        let mut encoder = GzEncoder::new(gz_file, Compression::default());
        std::io::copy(&mut stdout, &mut encoder)
            .map_err(|e| Error::new(format!("Failed to write '{}': {e}", dest.display())))?;
        encoder
            .finish()
            .map_err(|e| Error::new(format!("Failed to finish '{}': {e}", dest.display())))?;
        Ok(())
    })();
    // if our compress/write failed, kill the subprocess so one still writing to the pipe
    // we stopped draining can't linger
    if stream.is_err() {
        let _ = child.kill();
    }
    // always reap the child to avoid a zombie, then fold its exit into the result: our
    // own streaming error is the root cause and wins, otherwise a non-zero exit fails
    let status = child
        .wait()
        .map_err(|e| Error::new(format!("{} save failed: {e}", runtime())))?;
    let result = stream.and_then(|()| {
        if status.success() {
            Ok(())
        } else {
            Err(Error::new(format!("{} save failed for '{url}'", runtime())))
        }
    });
    // don't leave a truncated/garbage archive behind on failure; it would look importable
    if result.is_err() {
        let _ = std::fs::remove_file(dest);
    }
    result
}

/// Load an image from a tarball into the local cache (`<runtime> load -i <tar>`)
///
/// # Arguments
///
/// * `tar` - The path to the (optionally gzipped) image tarball
/// * `bar` - The progress bar to update with status
pub async fn load(tar: &Path, bar: &Bar) -> Result<(), Error> {
    // reflect the current step in the shared progress bar
    bar.set_message("Loading image");
    // pass the tarball as a separate arg (not formatted into the string) so paths with
    // spaces or odd bytes are forwarded verbatim; the runtime detects gzip itself
    let output = command(["load", "-i"])
        .arg(tar)
        .output()
        .await
        .map_err(|e| Error::new(format!("Failed to run {} load: {e}", runtime())))?;
    // turn a non-zero exit into an error carrying the runtime's stderr
    check(
        &output,
        &format!("{} load failed for '{}'", runtime(), tar.display()),
    )
}

/// Retag a local image (`<runtime> tag <src> <dst>`)
///
/// # Arguments
///
/// * `src` - The existing local image reference
/// * `dst` - The new reference to apply
/// * `bar` - The progress bar to update with status
pub async fn tag(src: &str, dst: &str, bar: &Bar) -> Result<(), Error> {
    // reflect the current step in the shared progress bar
    bar.set_message("Retagging image");
    // run `<runtime> tag <src> <dst>` to add the new reference to the already-local image
    let output = command(["tag", src, dst])
        .output()
        .await
        .map_err(|e| Error::new(format!("Failed to run {} tag: {e}", runtime())))?;
    // turn a non-zero exit into an error carrying the runtime's stderr
    check(
        &output,
        &format!("{} tag failed ('{src}' -> '{dst}')", runtime()),
    )
}

/// Push an image to a registry (`<runtime> push <url>`)
///
/// # Arguments
///
/// * `url` - The fully-qualified image url to push
/// * `bar` - The progress bar to update with status
pub async fn push(url: &str, bar: &Bar) -> Result<(), Error> {
    // reflect the current step in the shared progress bar
    bar.set_message("Pushing image");
    // run `<runtime> push <url>` to upload the local image to its registry
    let output = command(["push", url])
        .output()
        .await
        .map_err(|e| Error::new(format!("Failed to run {} push: {e}", runtime())))?;
    // turn a non-zero exit into an error carrying the runtime's stderr
    check(&output, &format!("{} push failed for '{url}'", runtime()))
}

// ─── Streaming variants ──────────────────────────────────────────────────────
//
// These forward the runtime's stdout/stderr straight to this process's terminal so
// the user sees build/push progress as it happens. They take no progress bar — a
// spinner would fight with the CLI's own live output — so callers should print
// their own framing around them.

/// Run a configured runtime command with its stdout/stderr inherited (streamed
/// to the terminal), returning an error describing `ctx` on a non-zero exit
///
/// # Arguments
///
/// * `cmd` - The configured runtime command to run
/// * `ctx` - A human-readable description of the attempt, used in the error
async fn run_streamed(mut cmd: Command, ctx: String) -> Result<(), Error> {
    // inherit both stdio streams so the runtime's live build/push output reaches the
    // user's terminal directly; `status` (not `output`) is used so nothing is captured
    let status = cmd
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .await
        .map_err(|e| Error::new(format!("Failed to run {}: {e}", runtime())))?;
    // a clean exit needs no error; the diagnostics already streamed to the terminal
    if status.success() {
        return Ok(());
    }
    // stderr was inherited, so build an error from the exit status alone, distinguishing
    // a normal non-zero exit from termination by a signal (no code available)
    Err(Error::new(match status.code() {
        Some(code) => format!("{ctx} ({} exited with code {code})", runtime()),
        None => format!("{ctx} ({} terminated by signal)", runtime()),
    }))
}

/// Optional runtime flags for a streamed build
#[derive(Clone, Copy, Default)]
pub struct BuildOptions {
    /// Pass `--no-cache` so the build ignores the layer cache
    pub no_cache: bool,
    /// Pass `--pull` so the runtime fetches a fresh copy of every referenced image
    /// (including the `FROM` base) instead of using a locally cached one
    pub pull: bool,
}

/// Assemble the `build` subcommand arguments (everything after the runtime binary), in order
///
/// Kept separate from [`build_streamed`] so the flag/arg assembly can be unit-tested without
/// spawning a runtime. The order is `build -t <tag> [--no-cache] [--pull]
/// [--build-arg k=v ...] <context>`.
///
/// # Arguments
///
/// * `tag` - The tag to build the image as
/// * `context` - The build context directory
/// * `build_args` - `(key, value)` pairs forwarded as `--build-arg key=value`
/// * `opts` - The optional runtime flags (`--no-cache` / `--pull`)
fn build_command_args(
    tag: &str,
    context: &Path,
    build_args: &[(String, String)],
    opts: BuildOptions,
) -> Vec<String> {
    let mut args = vec!["build".to_string(), "-t".to_string(), tag.to_string()];
    // ignore the layer cache when requested
    if opts.no_cache {
        args.push("--no-cache".to_string());
    }
    // force a fresh pull of referenced images when requested
    if opts.pull {
        args.push("--pull".to_string());
    }
    // forward each build arg as a `--build-arg key=value` pair
    for (key, value) in build_args {
        args.push("--build-arg".to_string());
        args.push(format!("{key}={value}"));
    }
    // the build context is always the final positional argument
    args.push(context.to_string_lossy().into_owned());
    args
}

/// Build an image, streaming the runtime's output to the terminal
///
/// # Arguments
///
/// * `tag` - The tag to build the image as
/// * `context` - The build context directory (must contain a Dockerfile)
/// * `build_args` - `(key, value)` pairs passed to the runtime as `--build-arg key=value`
/// * `opts` - The optional runtime flags (`--no-cache` / `--pull`)
pub async fn build_streamed(
    tag: &str,
    context: &Path,
    build_args: &[(String, String)],
    opts: BuildOptions,
) -> Result<(), Error> {
    // assemble `<runtime> build -t <tag> [--no-cache] [--pull] [--build-arg k=v ...] <context>`
    let cmd = command(build_command_args(tag, context, build_args, opts));
    // stream the build to the terminal; the context path is included in the error so a
    // failed build points at exactly which build directory was attempted
    run_streamed(
        cmd,
        format!(
            "{} build failed for '{tag}' ({})",
            runtime(),
            context.display()
        ),
    )
    .await
}

/// Retag a local image, streaming the runtime's output to the terminal
///
/// # Arguments
///
/// * `src` - The existing local image reference
/// * `dst` - The new reference to apply
pub async fn tag_streamed(src: &str, dst: &str) -> Result<(), Error> {
    // build `<runtime> tag <src> <dst>` to alias an extra reference onto a local image
    let cmd = command(["tag", src, dst]);
    // stream it so it shares the same live-output framing as the surrounding build/push
    run_streamed(cmd, format!("{} tag failed ('{src}' -> '{dst}')", runtime())).await
}

/// Push an image, streaming the runtime's output to the terminal
///
/// # Arguments
///
/// * `url` - The fully-qualified image url to push
pub async fn push_streamed(url: &str) -> Result<(), Error> {
    // build `<runtime> push <url>` to upload the local image to its registry
    let cmd = command(["push", url]);
    // stream it so push progress is visible live rather than buffered until completion
    run_streamed(cmd, format!("{} push failed for '{url}'", runtime())).await
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Each runtime maps to its expected CLI binary name
    #[test]
    fn binary_names_match_runtime() {
        // the binary names are what every command() invocation shells out to
        assert_eq!(ContainerRuntime::Docker.binary(), "docker");
        assert_eq!(ContainerRuntime::Podman.binary(), "podman");
    }
    /// An explicit flag wins over the config value and the detected runtime
    #[test]
    fn resolve_prefers_flag() {
        // flag, config, and detection all disagree so the result proves the flag wins
        let chosen = resolve(
            Some(ContainerRuntime::Podman),
            Some(ContainerRuntime::Docker),
            || ContainerRuntime::Docker,
        );
        assert_eq!(chosen, ContainerRuntime::Podman);
    }
    /// The config value is used when no flag is given
    #[test]
    fn resolve_falls_back_to_config() {
        // with no flag, the config value should beat the detection fallback
        let chosen = resolve(None, Some(ContainerRuntime::Podman), || ContainerRuntime::Docker);
        assert_eq!(chosen, ContainerRuntime::Podman);
    }
    /// The detection fallback is used when neither flag nor config is set
    #[test]
    fn resolve_falls_back_to_detected() {
        // with neither flag nor config, the detection closure's result is used
        let chosen = resolve(None, None, || ContainerRuntime::Podman);
        assert_eq!(chosen, ContainerRuntime::Podman);
    }
    /// The detection closure is not called when a flag or config value is set
    #[test]
    fn resolve_skips_detection_when_chosen() {
        // a panicking closure asserts detection (PATH probing) is never invoked once a
        // runtime is already chosen, which is what keeps detection lazy in production
        let chosen = resolve(None, Some(ContainerRuntime::Docker), || {
            panic!("detection should not run when a runtime is already chosen")
        });
        assert_eq!(chosen, ContainerRuntime::Docker);
    }

    /// A `(key, value)` pair as owned strings, for terse build-arg fixtures
    ///
    /// # Arguments
    ///
    /// * `key` - The build-arg key
    /// * `value` - The build-arg value
    fn arg(key: &str, value: &str) -> (String, String) {
        // build_command_args takes owned strings, so materialize them up front
        (key.to_string(), value.to_string())
    }

    /// With default options, no `--no-cache`/`--pull` are emitted and the context is last
    #[test]
    fn build_args_default_has_no_flags() {
        // default options must produce only the base build command with no extra flags
        let args = build_command_args("reg/x:1", Path::new("ctx"), &[], BuildOptions::default());
        assert_eq!(args, vec!["build", "-t", "reg/x:1", "ctx"]);
    }

    /// `--no-cache` alone is emitted (and not `--pull`)
    #[test]
    fn build_args_no_cache_only() {
        // only no_cache is set, so --no-cache appears but --pull must not
        let opts = BuildOptions { no_cache: true, pull: false };
        let args = build_command_args("reg/x:1", Path::new("ctx"), &[], opts);
        assert!(args.contains(&"--no-cache".to_string()));
        assert!(!args.contains(&"--pull".to_string()));
    }

    /// `--pull` alone is emitted (and not `--no-cache`)
    #[test]
    fn build_args_pull_only() {
        // only pull is set, so --pull appears but --no-cache must not
        let opts = BuildOptions { no_cache: false, pull: true };
        let args = build_command_args("reg/x:1", Path::new("ctx"), &[], opts);
        assert!(args.contains(&"--pull".to_string()));
        assert!(!args.contains(&"--no-cache".to_string()));
    }

    /// Both flags are emitted when both options are set
    #[test]
    fn build_args_both_flags() {
        // both options set, so both flags must be present
        let opts = BuildOptions { no_cache: true, pull: true };
        let args = build_command_args("reg/x:1", Path::new("ctx"), &[], opts);
        assert!(args.contains(&"--no-cache".to_string()));
        assert!(args.contains(&"--pull".to_string()));
    }

    /// Build args are rendered as `--build-arg key=value` and the context stays last
    #[test]
    fn build_args_render_build_args_then_context() {
        // two build args exercise both the rendering and the ordering guarantee
        let build_args = vec![arg("IMAGE", "ubuntu:22.04"), arg("VERSION", "1")];
        let args =
            build_command_args("reg/x:1", Path::new("ctx"), &build_args, BuildOptions::default());
        // join into one string so the `--build-arg key=value` pairing can be asserted
        let joined = args.join(" ");
        assert!(joined.contains("--build-arg IMAGE=ubuntu:22.04"));
        assert!(joined.contains("--build-arg VERSION=1"));
        // the context must remain the final positional argument after the build args
        assert_eq!(args.last().map(String::as_str), Some("ctx"));
    }
}
