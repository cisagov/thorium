FROM docker.io/seipharos/pharos:latest

ENTRYPOINT ["apianalyzer", "--batch", "--threads", "4", "--maximum-memory", "16384", "--allow-64bit", "--format", "json", "--out_file", "/tmp/thorium/results"]