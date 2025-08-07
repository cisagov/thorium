FROM docker.io/seipharos/pharos:latest

ENTRYPOINT ["callanalyzer", "--batch", "--threads", "4", "--maximum-memory", "16384", "--allow-64bit", "--allow-unknown", "--show-symbolic", "--json", "/tmp/thorium/results", "--pretty-json", "4"]