# Reactor
---
While we can rely on K8s to spawn workers that is not true on bare metal
systems or on Windows. To replicate this Thorium has the reactor. The
Thorium reactor periodically polls the Thorium API for information on
its node and spawns/despawns workers to match. This allows us to share
the same agent logic across all systems without making the agent more
complex.

## Reactor Types

Thorium supports several types of reactors:

### KVM Reactor
The KVM reactor manages virtual machines using libvirt/KVM. It's designed for Linux systems and provides isolated environments for analysis jobs.

**Key Features:**
- Manages pre-configured VMs using overlays of golden images
- Uses the QEMU guest agent to transfer and launch the Thorium agent
- Manages the full VM lifecycle, cleaning up resources on exit and error

**Setup Guide:** For detailed instructions on configuring VMs for the KVM reactor, see the [KVM VM Setup Guide](../admins/reactor_kvm_setup.md).

### Bare Metal Reactor
The bare metal reactor runs jobs directly on the host system without virtualization. This provides maximum performance but with less isolation.

### Windows Reactor
The Windows reactor manages Windows containers on Windows host systems.

## How the Reactor Works

1. **Polling**: The reactor periodically checks the Thorium API for the current state of its node
2. **Worker Management**: It compares the desired state with the current state and spawns/shuts down workers accordingly
3. **Lifecycle Management**: For each worker, the reactor handles the complete lifecycle from creation to cleanup
4. **Resource Tracking**: It monitors resource usage and reports status back to the Thorium API
