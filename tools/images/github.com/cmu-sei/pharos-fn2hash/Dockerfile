FROM docker.io/seipharos/pharos:latest

ENTRYPOINT ["fn2hash", "--batch", "--threads", "4", "--maximum-memory", "16384", "--allow-64bit", "--json", "/tmp/thorium/results"]