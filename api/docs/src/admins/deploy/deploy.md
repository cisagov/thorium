

# Deploy Thorium on Kubernetes (K8s)

> This documentation is for Thorium admins looking to deploy a new Thorium instance. This guide is
> just an example and you will need to modify these steps to make them to work in your
> environment. The instructions described below setup Thorium and it's dependencies on a blank K8s
> cluster that is hosted on servers or VMs. It does not use any specific cloud environment,
> however nothing precludes deployment of Thorium into the cloud.

## Prerequisites

You will need to deploy a working K8s cluster on baremetal servers, VMs, or within a hosted cloud
environment to start this guide. The K8s cluster will need to have a storage provisioner that
can provide persistent volume claims (PVCs) for the database and tracing services that Thorium
utilizes. Additionally, admins will need account credentials and permissions to create buckets
within an S3-compatible object storage interface that is accessible from the K8s cluster.

## Install Infrastructure Components

> For cloud deployments, you may skip the setup steps here for any database or other component that
> your cloud provider supports natively. Instead, you may choose to follow their guides for setup of
> the equivalent software stack.

#### Traefik (ingress proxy)

To deploy Traefik as an ingress proxy, follow these [installation steps](./install-traefik.md).

#### Rook (converged storage)

> This step is only required if your K8s cluster has attached storage that you wish to use to host
> S3-compatible and block device storage in a hyperconverged manner.

To deploy Rook, follow these [installation steps](./install-rook.md).

#### Redis

To deploy Redis, follow these [installation steps](./install-redis.md).

#### Scylla

To deploy Scylla, follow these [installation steps](./install-scylla.md).

#### Elastic

To deploy Elastic, follow these [installation steps](./install-elastic.md).

#### Tracing (Quickwit and Jaeger)

To deploy Quickwit and Jaeger, follow these [installation steps](./install-tracing.md).

## Deploy Thorium Operator and Cluster

The finals steps involve deploying the Thorium operator, a `ThoriumCluster` custom resource, and
Traefik `IngressRoutes` as described in the [Deploy Thorium](./deploy-thorium.md) section.
