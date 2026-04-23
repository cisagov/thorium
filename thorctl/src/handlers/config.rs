//! Handles config commands

use std::{collections::HashSet, path::PathBuf};

use thorium::{CtlConf, Error, client::conf::GitSettings};

use crate::args::{
    Args,
    config::{Config, ConfigOpts},
};

/// Update the Thorctl configuration, returning the modified [`CtlConf`]
///
/// # Arguments
///
/// * `config` - The config to modify
/// * `cmd` - The optional updates to the Thorctl config
fn update_config(mut config: CtlConf, opts: &ConfigOpts) -> CtlConf {
    // set opts if they were set
    if let Some(keys) = &opts.git_ssh_keys {
        config.git = Some(GitSettings::new(keys));
    }
    if let Some(invalid_certs) = opts.invalid_certs {
        config.client.invalid_certs = invalid_certs;
    }
    if let Some(invalid_hostnames) = opts.invalid_hostnames {
        config.client.invalid_hostnames = invalid_hostnames;
    }
    if opts.clear_certificate_authorities {
        config.client.certificate_authorities.clear();
    } else {
        let mut cert_set: HashSet<PathBuf> =
            HashSet::from_iter(config.client.certificate_authorities.clone());
        // append any new certificate authorities
        cert_set.extend(opts.certificate_authorities.clone());
        // remove certificate authorities
        for cert in &opts.remove_certificate_authorities {
            cert_set.remove(cert);
        }
        config.client.certificate_authorities = cert_set.into_iter().collect();
    }
    if let Some(timeout) = opts.timeout {
        config.client.timeout = timeout;
    }
    if let Some(skip_insecure_warning) = opts.skip_insecure_warning {
        config.skip_insecure_warning = Some(skip_insecure_warning);
    }
    if let Some(skip_update) = opts.skip_update {
        config.skip_update = Some(skip_update);
    }
    if let Some(default_editor) = &opts.default_editor {
        config.default_editor.clone_from(default_editor);
    }
    config
}

/// Modify the Thorctl configuration file given by `--config`
///
/// # Arguments
///
/// * `args` - The base Thorctl arguments
/// * `cmd` - The config command that was run
pub fn config(args: &Args, cmd: &Config) -> Result<(), Error> {
    // deserialize the Thorctl configuration file
    let Ok(thorctl_conf) = CtlConf::from_path(&args.config) else {
        return Err(Error::new(format!(
            "Missing or invalid config file at '{}': first login \
            using 'thorctl login' to generate a valid base configuration file",
            args.config.to_string_lossy()
        )));
    };
    // update the Thorctl config
    let new_conf = update_config(thorctl_conf, &cmd.config_opts);
    // write the new configuration file
    let conf_file = std::fs::File::create(&args.config)?;
    serde_norway::to_writer(conf_file, &new_conf)?;
    Ok(())
}
