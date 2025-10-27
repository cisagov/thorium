FROM docker.io/seipharos/pharos:latest

ENTRYPOINT ["callanalyzer", "--batch", "--maximum-memory", "32768", "--allow-64bit", "--allow-unknown", "--show-symbolic", "--json", "/tmp/thorium/results", "--pretty-json", "4"]