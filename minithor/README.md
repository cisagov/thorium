
# Overview

Minithor utilizes Minikube to provide a local Kubernetes instance with minimal custom configuration required. Such an instance is useful for development and testing of Thorium as well as small stand alone analyst fly-away kits where external network access may not be available. Minithor deployments are not highly available distributed systems like our production instances and provides minimal redundancy. The Thorium deployment produced by following these instructions should be considered Beta. We will work to improve its stability over time. While a Minithor deployment is accessible only from your localhost, it has not been configured to be secure. Please change DB passwords if working with sensitive data on a multi-user system using a Minithor deployment.

### Requirements and "Disclosures"

To deploy Minithor, you will need a container runtime such as that provided by the docker engine. Minithor also requires a relatively beefy machine, with > 12 GiB of memory, 8+ CPUs, and 100GiB of local storage.

### Deploy Minikube

Install and start minikube and any necessary plugins.

```bash
./install-linux
# or ./install-mac-m1
```

Add this to your environment settings after installation:

```bash
alias kubectl="minikube kubectl --"
```

### Create registry auth file

In the project directory you will need to create a file called `.dockerconfigjson` containing the authentication credentials for the user account/registry containing the thorium container image.

Create the `.dockerconfigjson` via the `docker login` command. The registry url must match that used by the images Thorium will run:

```bash
docker login registry.domain:port
```

The registry auth information will be structured like this:

```bash
cat .dockerconfigjson
{
	"auths": {
		"registry.domain:port": {
			"auth": "<base64 of username:token/password>"
		}
    }
}
```

Once this registry auth file has been created, copy the file (default path is `~/.docker/config.json` for most linux systems, must be manually created on mac) to the project directory and rename it to `.dockerconfigjson`.

### Deploy Dependencies

Thorium requires persistent storage interfaces a tracing API and an operator. Lets deploy these dependencies.

If your organization maintains a proxy for all traffic going to the internet, you will need to export proxy settings such as the following:
```bash
cat proxy

#!/bin/bash
export HTTP_PROXY=<HTTP_PROXY_URL:PORT>
export HTTPS_PROXY=<HTTPS_PROXY_URL:PORT>
export NO_PROXY=localhost,127.0.0.1,10.0.0.0/8,192.168.0.0/16
```

Once you have built that proxy file it can be reused in different terminal windows with:

```bash
source proxy
```

Alternatively, those proxy settings can be added into your shell's settings file.

Now deploy the dependencies:

```bash
./deploy
```

### Deploy Thorium

```bash
kubectl create -n thorium -f thorium-cluster.yml
```

### Set Password For Node's Docker User

You only have to do this once and only when using privileged ports for your local host port mapping. Keep track of the docker-in-docker password you set so you can tunnel to the Thorium UI/API later.

```bash
minikube ssh
sudo su -
passwd docker
# New password: 
# Retype new password: 
# passwd: password updated successfully
exit
exit
```

### Setup Tunnel (when using Thorium)

This is a blocking command that can must be run in a dedicated terminal window or put in the background.

```bash
minikube tunnel
# or ./expose
```

### Setup Dev Tunnels (Elastic/Kibana, Scylla, Redis)

This is a blocking command that can must be run in a dedicated terminal window or put in the background.

```bash
./expose-dev
```

### Get Thorium admin password

```bash
kubectl get secret -n thorium thorium-pass --template={{.data.thorium}} | base64 --decode; echo
```

### Cleanup of Minithor

```bash
./stop
./delete
rm -r ~/.minikube ~/.kube
```
