
# Overview

Minithor utilizes Minikube to provide a local Kubernetes instance with minimal custom configuration required. Such an instance is useful for development and testing of Thorium as well as small stand alone analyst fly-away kits where external network access may not be available. Minithor deployments are not highly available distributed systems like our production instances and provides minimal redundancy. The Thorium deployment produced by following these instructions should be considered Beta. We will work to improve its stability over time. While a Minithor deployment is accessible only from your localhost, all backing-service passwords are randomly generated per deploy. The default admin user password can be randomized with `--rand-password`.

### Requirements

To deploy Minithor, you will need a container runtime such as that provided by the docker engine or podman. KVM/libvirt (`kvm2` driver) is also supported on Linux. Minithor automatically selects the best available driver in order of preference: podman > docker > kvm2. Minithor also requires a relatively beefy machine, with 16+ GiB of memory, 8+ CPUs, and 100GiB of local storage.

## Usage

All lifecycle operations are available through the `minithor` CLI. The script reads input files from your current working directory by default and can be installed globally (e.g. `/usr/local/bin/minithor`).

```
minithor — manage a local Thorium instance on Minikube

Usage: minithor <command> [options]

Commands:
  minikube         Manage the minikube cluster (install, delete)
  start            Start a previously stopped minikube cluster
  deploy           Deploy all Thorium services and backing infrastructure
  get-config       Extract the running Thorium config to ~/thorium.yml
  expose           Port-forward Thorium (and optionally backing services) to localhost
  stop             Stop the minikube cluster (preserves state)
  cleanup          Remove all Thorium resources for a fresh deploy (requires --confirm)

Global options:
  -h, --help       Show this help message

Run 'minithor <command> --help' for command-specific options.
```

### Working directory layout

When running `minithor` commands, the following files are looked for in your current working directory:

| File | Used by | Required? |
|------|---------|-----------|
| `.certs/*.crt` | `minikube install` | No — optional TLS proxy certs to trust in minikube |
| `thorium-cluster.yml` | `deploy` | No — cluster config (a built-in default is used if absent) |
| `.dockerconfigjson` | `deploy` | No — private registry credentials for pulling images |
| `banner.txt` | `deploy` | No — login banner (a default is generated if absent) |

All paths can be overridden with CLI flags — run `minithor <command> --help` for details.

### Install Minikube

Install and start minikube and any necessary plugins.

```bash
minithor minikube install [--cpus <n>] [--memory <n>] [--certs-dir <path>]
```

| Flag | Description |
|------|-------------|
| `--cpus <n>` | Number of CPUs to allocate to minikube (default: 8) |
| `--memory <n>` | Memory in GiB to allocate to minikube (default: 16) |
| `--certs-dir <path>` | Directory containing `.crt` files to trust (default: `$PWD/.certs/`) |

### Create registry auth file (optional)

If the Thorium container image is hosted in a private registry, create a `.dockerconfigjson` file in your working directory containing the registry credentials. The deploy command will detect this file and create a Kubernetes image pull secret automatically.

```bash
docker login registry.domain:port
cp ~/.docker/config.json .dockerconfigjson
```

If omitted, the operator will pull images without authentication (works for public registries like `ghcr.io`).

### Proxy configuration (optional)

If your organization maintains a proxy for all traffic going to the internet, export proxy settings before running the deploy command:

```bash
export HTTP_PROXY=<HTTP_PROXY_URL:PORT>
export HTTPS_PROXY=<HTTPS_PROXY_URL:PORT>
export NO_PROXY=localhost,127.0.0.1,10.0.0.0/8,192.168.0.0/16
```

Or use the provided proxy file: `source proxy`

### Deploy

The `deploy` command handles the full deployment in a single step: all backing services (Redis, Elasticsearch, ScyllaDB, MinIO, Postgres, Quickwit, Jaeger), the Thorium operator, the ThoriumCluster resource, and a default test user.

```bash
minithor deploy [options]
```

| Flag | Description |
|------|-------------|
| `--config <path>` | Path to `thorium-cluster.yml` (default: `$PWD/thorium-cluster.yml`; a built-in default is used if absent) |
| `--docker-config <path>` | Path to `.dockerconfigjson` (default: `$PWD/.dockerconfigjson`) |
| `--banner <path>` | Path to `banner.txt` (default: `$PWD/banner.txt`, generates default if absent) |
| `--bin <path>` | Directory for downloaded binaries like thorctl (default: `/usr/local/bin`) |
| `--toolbox <path>` | Download location for `toolbox.json` (default: `$PWD/toolbox.json`) |
| `--user <name>` | Username for the initial admin user (default: `test`) |
| `--password <pass>` | Password for the initial admin user (default: `INSECURE_DEV_PASSWORD`) |
| `--rand-password` | Generate a random password for the admin user |
| `--registry` | Deploy a container registry (registry:2) in the thorium namespace with persistent storage |
| `--registry-user <name>` | Enable registry basic auth for this user (implies `--registry`, password is auto-generated and printed to stdout) |

This will:

1. Wait for the minikube cluster to be healthy
2. Install Helm and add required chart repos
3. Deploy Redis, Elasticsearch (ECK), cert-manager, ScyllaDB, MinIO, Jaeger, Kubegres, and Quickwit
4. Configure databases (Scylla roles/keyspace, Elasticsearch index/user, MinIO buckets)
5. Deploy the Thorium operator and create the ThoriumCluster CRD
6. Wait for all Thorium components (API, scaler, event-handler, search-streamer) to be running
7. Create an admin user (default: `test` / `INSECURE_DEV_PASSWORD`, customizable via `--user` / `--password` / `--rand-password`)
8. Install `thorctl` to the bin directory and import the default toolbox
9. Create a `static` group and an `allow-all` network policy
10. Optionally deploy a container registry with persistent storage (if `--registry` or `--registry-user` is specified)

All backing-service passwords (Redis, Scylla, Elasticsearch, MinIO, Postgres, and the Thorium API secret key) are randomly generated per deploy using a cryptographically secure RNG. These passwords are not displayed during deployment but can be retrieved afterward with `minithor get-config`.

To customize the ThoriumCluster configuration, provide your own config file:

```bash
minithor deploy --config path/to/thorium-cluster.yml
```

### Access Thorium

Port-forward the Thorium API to localhost:

```bash
minithor expose [--dev] [--port <port>] [--stop] [--status]
```

| Flag             | Description                                                     |
|------------------|-----------------------------------------------------------------|
| `--dev`          | Also forward database ports (Elastic, Kibana, Redis, MinIO, Scylla) |
| `--port <port>`  | Local port for the Thorium API (default: 8080)                  |
| `--stop`         | Stop all running port-forwards                                  |
| `--status`       | Show which port-forwards are running                            |

The registry service (port 5000) is also forwarded by default when a registry has been deployed. Port-forward logs are written to `/tmp/minithor-expose-logs/`.

Then open http://localhost:8080 in your browser and log in with the credentials shown at the end of the deploy output (default: `test` / `INSECURE_DEV_PASSWORD`).

### Get Config

Extract the running Thorium config from the cluster:

```bash
minithor get-config
# Writes thorium.yml to ~/thorium.yml
```

### Stop

Stop the minikube cluster while preserving its state:

```bash
minithor stop
```

Resume later with `minithor start`.

### Start

Restart a previously stopped cluster:

```bash
minithor start
```

### Cleanup

Remove all deployed Thorium resources and backing services for a fresh deploy. The `--confirm` flag is required:

```bash
minithor cleanup --confirm
```

This removes all namespaces, Helm releases, CRDs, and cluster-level RBAC created by the deploy step. Individual step failures are logged but do not abort the cleanup. The minikube cluster itself is preserved.

### Delete Minikube

Completely remove minikube, its data, and associated binaries. The `--confirm` flag is required:

```bash
minithor minikube delete --confirm
```

This runs `minikube delete --all --purge`, removes `~/.minikube` and `~/.kube`, uninstalls the minikube and kubectl binaries, and prunes unused container images. This is irreversible.
