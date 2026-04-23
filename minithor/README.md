
# Overview

Minithor utilizes Minikube to provide a local Kubernetes instance with minimal custom configuration required. Such an instance is useful for development and testing of Thorium as well as small stand alone analyst fly-away kits where external network access may not be available. Minithor deployments are not highly available distributed systems like our production instances and provides minimal redundancy. The Thorium deployment produced by following these instructions should be considered Beta. We will work to improve its stability over time. While a Minithor deployment is accessible only from your localhost, it has not been configured to be secure. Please change DB passwords if working with sensitive data on a multi-user system using a Minithor deployment.

### Requirements

To deploy Minithor, you will need a container runtime such as that provided by the docker engine or podman. Minithor also requires a relatively beefy machine, with > 12 GiB of memory, 8+ CPUs, and 100GiB of local storage.

### Install Minikube

Install and start minikube and any necessary plugins.

```bash
./install
```

### Create registry auth file (optional)

If the Thorium container image is hosted in a private registry, create a `.dockerconfigjson` file in this directory containing the registry credentials. The deploy script will detect this file and create a Kubernetes image pull secret automatically.

```bash
docker login registry.domain:port
cp ~/.docker/config.json .dockerconfigjson
```

If omitted, the operator will pull images without authentication (works for public registries like `ghcr.io`).

### Proxy configuration (optional)

If your organization maintains a proxy for all traffic going to the internet, export proxy settings before running the deploy script:

```bash
export HTTP_PROXY=<HTTP_PROXY_URL:PORT>
export HTTPS_PROXY=<HTTPS_PROXY_URL:PORT>
export NO_PROXY=localhost,127.0.0.1,10.0.0.0/8,192.168.0.0/16
```

Or use the provided proxy file: `source proxy`

### Deploy

The `deploy` script handles the full deployment in a single step: all backing services (Redis, Elasticsearch, ScyllaDB, MinIO, Quickwit, Jaeger), the Thorium operator, the ThoriumCluster resource, and a default test user.

```bash
./deploy
```

This will:

1. Wait for the minikube cluster to be healthy
2. Install Helm and add required chart repos
3. Deploy Redis, Elasticsearch (ECK), cert-manager, ScyllaDB, MinIO, Jaeger, Kubegres, and Quickwit
4. Configure databases (Scylla roles/keyspace, Elasticsearch index/user, MinIO buckets)
5. Deploy the Thorium operator and create the ThoriumCluster CRD
6. Wait for all Thorium components (API, scaler, event-handler, search-streamer) to be running
7. Create a test admin user (`test` / `INSECURE_DEV_PASSWORD`)
8. Install `thorctl` from the API and import the default toolbox (analysis tools and pipelines)

To customize the ThoriumCluster configuration, copy the example and edit it before running the deploy script:

```bash
cp thorium-cluster.yml.example thorium-cluster.yml
# edit thorium-cluster.yml as needed
./deploy.sh
```

If no `thorium-cluster.yml` exists, the script falls back to `thorium-cluster.yml.example`.

### Access Thorium

Start the minikube tunnel in a separate terminal (this is a blocking command):

```bash
./expose --help
Usage: ./expose [--dev] [--stop] [--status]
  --dev     Also forward database ports (Elastic, Kibana, Redis, MinIO, Scylla)
  --stop    Stop all running port-forwards
  --status  Show which port-forwards are running
```

Then open http://localhost in your browser and log in:
- **Username:** `test`
- **Password:** `INSECURE_DEV_PASSWORD`

### Dev tunnels (Elastic/Kibana, Scylla, Redis)

To access backing services directly from your host (useful for debugging):

```bash
./expose --dev
```

### Cleanup

Remove all deployed resources (without deleting the minikube cluster itself):

```bash
./cleanup
```

To fully remove minikube:

```bash
./stop
./delete
rm -r ~/.minikube ~/.kube
```
