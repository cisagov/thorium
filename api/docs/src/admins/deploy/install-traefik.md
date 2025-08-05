
## Deploy Traefik 

Traefik is a reverse proxy and load balancer that enables routing of http and https traefik to
Thorium and any other web services you deploy in K8s (such as a local container registry).

### 1) Install the latest helm repo for Traefik

```bash
helm repo add traefik https://helm.traefik.io/traefik
helm repo update
```

### 2) Get a default values file for a Traefik release

```bash
helm show values traefik/traefik > traefik-values.yml
```

### 3) Modify the default helm values for Traefik

Update read and write response timeouts for http and https requests going through the traefik ingress proxy.

```yaml
ports:
  ...
  web:
    ...
    transport:
      respondingTimeouts:
        readTimeout:   0 # @schema type:[string, integer, 0]
        writeTimeout:  0 # @schema type:[string, integer, 0]
        idleTimeout:   600 # @schema type:[string, integer, 600]
  ...
  ...
  websecure:
    ...
    transport:
      respondingTimeouts:
        readTimeout:   0 # @schema type:[string, integer, 0]
        writeTimeout:  0 # @schema type:[string, integer, 0]
        idleTimeout:   600 # @schema type:[string, integer, 600]
```

Update the IP addresses for web traffic that will access your Thorium instances from locations external to K8s.

```yaml
service:
  ...
  externalIPs:
    - 1.2.3.4
    - 1.2.3.5
    - 1.2.3.6
    - 4.3.2.1
```

Explicitly disable anonymous usage reporting for networked Traefik deployments.
```yaml
globalArguments:
...
- "--global.sendanonymoususage=false"
```

### 4) Create a namespace for Traefik and deploy

```bash
kubectl create ns traefik
sleep 5
helm install -f traefik-values.yml traefik traefik/traefik --namespace=traefik
```

You can update the values of an existing Traefik helm chart with the following command:

```bash
helm upgrade -f traefik-values.yml --namespace=traefik traefik traefik/traefik
```

### 5) Verify the Traefik pod started

```bash
kubectl get pods -n traefik
# NAME                       READY   STATUS    RESTARTS   AGE
# traefik-HASH               1/1     Running   0          1h
```