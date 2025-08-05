
## Deploy Thorium

### 1) Deploy Thorium Operator

#### Create Thorium ServiceAccount and RBAC

The Thorium operator and scaler can be configured to use a service account with the ability to
modify K8s resources. This is the default configuration for single K8s cluster Thorium deployments.

Start by creating a namespace for all Thorium resources.

```bash
kubectl create ns thorium
```

Create a Thorium ServiceAccount and roles

```bash,editable
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: thorium
  namespace: thorium
imagePullSecrets:
  - name: registry-token
automountServiceAccountToken: true
---

apiVersion: v1
kind: Secret
metadata:
  name: thorium-account-token
  namespace: thorium
  annotations:
    kubernetes.io/service-account.name: thorium
type: kubernetes.io/service-account-token
---

apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  # "namespace" omitted since ClusterRoles are not namespaced
  name: thorium-operator
rules:
### https://kubernetes.io/docs/reference/kubectl/#resource-types
### create custom resources
- apiGroups: ["apiextensions.k8s.io"] 
  resources: ["customresourcedefinitions"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
### any custom resources under the sandia.gov group
- apiGroups: ["sandia.gov"] 
  resources: ["*"]
  verbs: ["*"]
### deployments
- apiGroups: ["apps"] 
  resources: ["deployments"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
### networking
- apiGroups: ["networking.k8s.io"] 
  resources: ["networkpolicies"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
### events
- apiGroups: ["events.k8s.io"] 
  resources: ["events"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
### v1 API resources
- apiGroups: [""] 
  resources: ["pods", "services", "secrets", "configmaps", "nodes", "namespaces", "events"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---

apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: thorium-operator-binding
subjects:
- kind: ServiceAccount
  name: thorium 
  namespace: thorium
  #apiGroup: "rbac.authorization.k8s.io"
roleRef:
  kind: ClusterRole
  name: thorium-operator
  apiGroup: "rbac.authorization.k8s.io" 
EOF
```

#### Create a registry pull secret (optional)

Create a registry token that will enable pulling the Thorium container image from the registry.

```bash
kubectl create secret generic operator-registry-token --namespace="thorium" --type=kubernetes.io/dockerconfigjson --from-file=".dockerconfigjson"
```

Here is an example of a `.dockerconfigjson` file. Replace the fields wrapped by `<>` with registry values.

```json
{
	"auths": {
		"<REGISTRY.DOMAIN>": {
			"auth": "<base64 of username:password>"
		}
	}
}
```

#### Create the Thorium Operator

Update the `image` field with the correct registry path and tag for the Thorium container image.

```bash,editable
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: operator
  namespace: thorium
  labels:
    app: operator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: operator
  template:
    metadata:
      labels:
          app: operator
    spec:
      serviceAccountName: thorium
      automountServiceAccountToken: true
      containers:
        - name: operator
          image: "<REGISTRY.DOMAIN/path/to/image/thorium:tag>"
          imagePullPolicy: Always
          resources:
            requests:
              memory: "1Gi"
              cpu: 1
            limits:
              memory: "1Gi"
              cpu: 1
          env:
            - name: "noproxy"
            - name: "http_proxy"
            - name: "https_proxy"
            - name: "NOPROXY"
            - name: "HTTP_PROXY"
            - name: "HTTPS_PROXY"
      imagePullSecrets:
        - name: operator-registry-token
EOF
```

#### Verify the operator has successfully started

```bash
kubectl rollout status --watch --timeout=600s deployment.apps/operator -n thorium
```

### 4) Create a Thorium banner ConfigMap

Create a text file called `banner.txt` that contains a banner message. This message will be displayed
when users login into the Thorium web interface.

```bash
kubectl create cm banner --from-file=/path/to/banner.txt -n thorium
```

### 5) Create a ThoriumCluster resource

The ThoriumCluster CRD defines database client access, Thorium cluster nodes, and much more. Enter
all the passwords, DB/S3 endpoints, and Thorium container image path/tag into this file. The operator
will use this the CRD to deploy a working Thorium instance. If this resource definition is updated
after the initial deployment the operator will role those changes out and restart Thorium components
such as the scaler and API.

#### Create a thorium-cluster.yml file and update the example values:

```yaml
apiVersion: sandia.gov/v1
kind: ThoriumCluster
metadata:
  name: prod
  namespace: thorium
spec:
  registry: "<REGISTRY.DOMAIN/path/to/image/thorium>"
  version: "<IMAGE TAG>"
  image_pull_policy: Always
  components:
    api:
      replicas: 1
      urls:
      - "<THORIUM FQDN>"
      ports:
      - 80
      - 443
    scaler:
      service_account: true
    baremetal_scaler: {}
    search_streamer: {}
    event_handler: {}
  config: 
    thorium:
      secret_key: "<SECRET>" 
      tracing:
        external:
          Grpc:
            endpoint: "http://quickwit-indexer.quickwit.svc.cluster.local:7281"
            level: "Info"
        local:
          level: "Info"
      files:
        bucket: "thorium-files"
        password: "SecretCornIsBest"
        earliest: 1610596807
      results:
        bucket: "thorium-result-files"
        earliest: 1610596807
      attachments:
        bucket: "thorium-comment-files"
      repos:
        bucket: "thorium-repo-files"
      ephemeral:
        bucket: "thorium-ephemeral-files"
      s3:
        access_key: "<KEY>"
        secret_token: "<TOKEN>"
        endpoint: "https://<S3 FQDN>"
      auth:
        local_user_ids:
          group: 1879048192
          user: 1879048192
        token_expire: 90
      scaler:
        crane:
          insecure: true
        k8s:
          clusters:
            kubernetes-admin@cluster.local:
              alias: "production"
              nodes:
                - "<K8s host 1>"
                - "<K8s host 2>"
                - "<K8s host 3>"
                - "<K8s host 4>"
                - "<K8s host 5>"
    redis:
      host: "redis.redis.svc.cluster.local"
      port: 6379
      password: "<PASSWORD>"
    scylla:
      nodes:
	- <SCYLLA IP 1>
	- <SCYLLA IP 2>
	- <SCYLLA IP 3>
      replication: 3
      auth:
        username: "thorium"
        password: "<PASSWORD>"
    elastic:
      node: "https://elastic-es-http.elastic-system.svc.cluster.local:9200"
      username: "thorium"
      password: "<PASSWORD>"
      results: "results"
  registry_auth:
    <REGISTRY.DOMAIN: <base64 USERNAME:PASSWORD>
    <REGISTRY2.DOMAIN: <base64 USERNAME:PASSWORD>
```

Thorium deployments that consist of multiple K8s clusters (managed by a single scaler pod) will
require a dedicated `kubeconfig` secret rather than the use of a service account that is default
for single cluster instances. This secret file must be built manually from the `kubeconfig` files
of the Thorium clusters that will be managed. The `service_account` field in the ThoriumCluster
CRD will be set to `false` for multi-cluster Thorium deployments. Most Thorium deployments will
are not multi-cluster.

#### Create the ThoriumCluster resource:

The operator will attempt to deploy the ThoriumCluster from the CRD you applied. This will include
creating secrets such as the shared thorium config (`thorium.yml`). It will also deploy scaler,
api, event-handler, and search-streamer pods if those have been specified.

```bash
# create the thorium CRD
kubectl create -f thorium-cluster-<DEPLOYMENT>.yml
```

### 6) Create IngressRoutes

IngressRoutes will be needed to direct web traffic to the Thorium API through the Traefik ingress
proxy. Modify the following command with the correct `THORIUM.DOMAIN` FQDN. A TLS certificate
called `api-certs` will be required. Without that K8s secret, Traefik will serve a default
self-signed cert that web clients will flag as insecure.

#### Create TLS K8s secret

Once you have signed `tls.crt` and `tls.key` files, create the `api-certs` secret using `kubectl`.

```bash
kubectl create secret tls api-certs --namespace="thorium" --key="tls.key" --cert="tls.crt"
```

#### Create Traefik IngressRoutes and Middleware

```yaml
apiVersion: traefik.io/v1alpha1
kind: TLSStore
metadata:
  name: default
  namespace: thorium
spec:
  defaultCertificate:
    secretName: api-certs
---

apiVersion: traefik.io/v1alpha1 
kind: Middleware
metadata:
  name: ui-prefix-prepend
spec:
  addPrefix:
    prefix: /ui
---

apiVersion: traefik.io/v1alpha1
kind: IngressRoute
metadata:
  name: thorium-ingress
spec:
  entryPoints:
    - websecure
  routes:
  - match: "Host(`THORIUM.DOMAIN`) && PathPrefix(`/api`)"
    kind: Rule
    services:
    - name: thorium-api
      port: 80
  - match: "Host(`THORIUM.DOMAIN`) && PathPrefix(`/assets`)"
    kind: Rule
    services:
    - name: thorium-api
      port: 80
  - match: "Host(`THORIUM.DOMAIN) && PathPrefix(`/ui`)"
    kind: Rule
    services:
    - name: thorium-api
      port: 80
  - match: "Host(`THORIUM.DOMAIN`)"
    kind: Rule
    services:
    - name: thorium-api
      port: 80
    middlewares:
      - name: ui-prefix-prepend
  tls:
    secretName: api-certs
```