use crate::provision;
use crate::{args::ProvisionSubCommands, Error};

/// Provision Thorium resources such as worker nodes
pub async fn handle(cmd: &ProvisionSubCommands) -> Result<(), Error> {
    // handle provisioning Thorium resources
    match cmd {
        ProvisionSubCommands::Node(node_args) => {
            // provision k8s servers by default
            if node_args.baremetal == true {
                println!("Provisioning baremetal servers not yet supported");
            } else {
                provision::nodes::conf_thorium_dir(&node_args.keys).await?;
            }
        }
    }
    Ok(())
}
