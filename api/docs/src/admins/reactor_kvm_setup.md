# Setting Up KVM VMs for Thorium Reactor

This guide explains how to properly configure KVM virtual machines for the Thorium reactor to manage and use as golden images.

## Overview

The Thorium reactor manages KVM-based virtual machines for running analysis jobs. To use this functionality, administrators must pre-configure VMs that the reactor can discover and manage.

## Prerequisites

- Libvirt/KVM installed and running
- `virsh` command-line tool available
- Appropriate VM disk images (QCOW2 format recommended)
- Basic understanding of libvirt domain management

### 2. Golden Snapshot

Each VM must have a snapshot named "golden" that represents a clean, ready-to-use state:
```bash
virsh snapshot-create-as thorium_windows_win10 golden
```

### 3. QEMU Guest Agent

All VMs for the Thorium reactor must have the QEMU guest agent installed and configured to
run at startup so the reactor can transfer and launch the Thorium agent. Be sure to include
this in your XML:

```xml
    <channel type='unix'>
      <target type='virtio' name='org.qemu.guest_agent.0' />
      <address type='virtio-serial' controller='0' bus='0' port='1' />
    </channel>
```

### 4. Agent Requirements

The VM should have:
- Network connectivity to the Thorium cluster
- Any required analysis tools pre-installed

## Setup Process

### Step 1: Prepare Your VM Disk Image

Create or obtain a QCOW2 disk image with your desired OS

### Step 2: Find/Create a Template Domain XML Configuration

Create an XML file with your VM configuration. The file type should be `qcow2`:

```xml
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2' />
      <source file='<YOUR-FILE-HERE>' />
      <target dev='vda' bus='virtio' />
      <address type='pci' domain='0x0000' bus='0x03' slot='0x00' function='0x0' />
    </disk>
```

Be sure to include a `virtio-serial` channel for the QEMU guest agent:

```xml
    <channel type='unix'>
      <target type='virtio' name='org.qemu.guest_agent.0' />
      <address type='virtio-serial' controller='0' bus='0' port='1' />
    </channel>
```

And VNC is probably helpful as well:

```xml
    <memballoon model='virtio'>
      <address type='pci' domain='0x0000' bus='0x05' slot='0x00' function='0x0'/>
    </memballoon>
```

### Step 3: Define the Domain

```bash
virsh define /path/to/your/xml.xml
```

### Step 4: Start and Configure the VM

1. Start the VM:
```bash
virsh start thorium_<GROUP>_<IMAGE>
```

2. Connect to the VM (using VNC or console) and:
   - Install the operating system
   - Install virtio drivers and the QEMU guest agent
   - Enable the QEMU guest agent to run at startup
   - Install any required analysis tools/scripts

3. Shut down the VM cleanly:

Through ACPI:

```bash
virsh shutdown thorium_<GROUP>_<IMAGE>
```

Or manually with VNC, console, or SSH.

### Step 5: Copy to a Read-only Golden Image

Now that your image is ready, you should copy it to a new file to serve as your golden image.

```bash
qemu-img convert -O qcow2 <IMAGE>.qcow2 <GOLDEN_IMAGE>.qcow2
```

### Step 6: Create a Golden Template XML

Replace hard-coded values for name, CPU, memory, and file with placeholders for
Thorium to dynamically update. If you dump the configuration with virsh dumpxml,
be sure to remove any domain-specific details like MAC addresses, IP addresses, etc.

```XML
<domain type='kvm'>
  <name>{NAME}<GROUP>_<IMAGE></name>
  <vcpu placement='static'>{CPU}</vcpu>
  <memory unit='MiB'>{MEMORY}</memory>
  ...
  <devices>
    <emulator>/usr/bin/qemu-system-x86_64</emulator>
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2'/>
      <source file=<'{FILE}'>/>
  ...
```

### Step 7: Move the Golden Image and XML to the final location

Place the golden image and XML together in a directory structured `<GOLDEN_BASE>/<GROUP>/<IMAGE>`, like
so:

```bash
# for image "test" in group "corn" the files should look something like this
find /base/golden
/base/golden/corn/test/test.qcow2
/base/golden/corn/test/test.xml
```

## How the Reactor Manages VMs

### Discovery Process

1. The reactor connects to libvirt using the socket specified in its configuration (default: `qemu:///system`)
2. During its check cycle, it lists all domains (VMs) from libvirt
3. It identifies Thorium-managed VMs by the naming pattern `thorium_*`
4. It attempts to query for the Thorium agent's status using the QEMU guest agent to manage the
  VMs' lifecycle

### Launch Process

1. **Find Available VM**: The reactor searches the golden directory for a disk image and XML
  matching the requested group/image
2. **Convert to Worker**: The reactor:
   - Creates a sparse overlay file from the golden image
   - Reads the XML file, replacing placeholders with Thorium image configuration values
   - Defines the VM with the modified XML
   - Starts the VM
3. **Launch Agent**: Uses the QEMU guest agent to transfer and launch the Thorium agent
4. **Monitor**: Tracks the VM's status and manages its lifecycle

## Storage Considerations

### Golden Image/XML Location

Specify `--golden-dir` to direct the reactor to where your golden images are stored.
If not set, the default is based on the set `--base_dir`, i.e. `BASE_DIR/golden`.
The structure should be like so:

```bash
# for image "test" in group "corn" the files should look something like this
find /base/golden
/base/golden/corn/test/test.qcow2
/base/golden/corn/test/test.xml
```

### Temporary Files

The reactor creates temporary files in the location specified by the `--temp-dir` argument.
If not set, the default is based on the set `--base_dir`, i.e. `BASE_DIR/tmp`. The
temp directory contains overlay qcow2 files for running domains. These are automatically
cleaned up when the domains are shut down.

## Troubleshooting

### VM Not Being Discovered

1. **Check Naming**: Ensure the VM name follows `thorium_{group}_{image}` pattern
2. **Verify Libvirt Connection**: Check the reactor's libvirt socket configuration
3. **List All Domains**:
```bash
virsh list --all
```

### QEMU Guest Agent Problems

1. The reactor *requires* the image has the QEMU guest agent installed
  and properly set up. Check if it's installed.

### Permission Issues

1. **Check libvirt group membership**:
```bash
usermod -a -G libvirt thorium-user
```

2. **Verify QCOW2 file permissions**

## Resource Allocation Strategy

### CPU Allocation

The reactor converts millicpu (mCPU) requests to vCPU using **round-up** logic:

- **Formula**: `max(1, ceil(mCPU / 1000))`
- **Examples**:
  - 1-1000 mCPU → 1 vCPU
  - 1001-2000 mCPU → 2 vCPU
  - 2001-3000 mCPU → 3 vCPU

**Important Behavior**: Due to rounding UP, the allocated vCPU may exceed the requested mCPU:
- 1001 mCPU requests → 2 vCPU allocated (2000 mCPU capacity)
- 1500 mCPU requests → 2 vCPU allocated (2000 mCPU capacity)

### Memory Allocation

The reactor follows [libvirt's defaults](https://libvirt.org/formatdomain.html#memory-allocation) for memory units.
The reactor attempts to detect units following the memory specification, and falls back to `KiB` as a default.

All of these are valid:

```xml
<memory unit='KiB'>{worker.resources.memory}</memory>
```

```xml
<memory unit='GB'>{worker.resources.memory}</memory>
```

When no units are specified, the units are translated in `KiB`:

```xml
<memory>{worker.resources.memory}</memory>
```

### Complete Example

For a worker requesting 1500 mCPU and 4096 MiB memory:

```xml
<domain type='kvm'>
  <name>worker_example</name>
  <vcpu>2</vcpu>  <!-- Rounded UP from 1500 mCPU -->
  <memory unit='MiB'>4096</memory>
  <!-- Rest of configuration... -->
</domain>
```

**Resource Impact**: This worker will consume 2 vCPU worth of capacity (2000 mCPU) due to rounding, even though only 1500 mCPU was requested.
