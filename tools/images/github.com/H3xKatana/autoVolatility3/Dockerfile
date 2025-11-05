ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

# avoid interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt

# 1) install system packages required to build some Volatility extras
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git build-essential gcc make pkg-config wget ca-certificates \
      unzip xz-utils python3-pip python3-dev libssl-dev libbz2-dev liblzma-dev \
      zlib1g-dev libffi-dev libsnappy-dev libzstd-dev libcapstone-dev \
      libyara-dev libpcre3-dev && \
    rm -rf /var/lib/apt/lists/*

# 2) upgrade pip & tooling
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 3) Clone volatility3 and install (editable install with dev extras)
RUN git clone --depth 1 https://github.com/volatilityfoundation/volatility3.git /opt/volatility3 && \
    cd /opt/volatility3 && \
    python3 -m pip install --no-cache-dir -e ".[dev]"

# 4) Download and extract the official Volatility3 symbol packs (Windows, Linux, mac)
#    (these are large; including them in the image makes the image size grow)
RUN mkdir -p /opt/volatility3/volatility3/symbols && \
    cd /opt/volatility3/volatility3/symbols && \
    wget -q -O windows.zip https://downloads.volatilityfoundation.org/volatility3/symbols/windows.zip && \
    wget -q -O linux.zip   https://downloads.volatilityfoundation.org/volatility3/symbols/linux.zip && \
    wget -q -O mac.zip     https://downloads.volatilityfoundation.org/volatility3/symbols/mac.zip && \
    unzip -q windows.zip && unzip -q linux.zip && unzip -q mac.zip && \
    rm -f windows.zip linux.zip mac.zip

# 5) Clone the autoVolatility3 script into the image
RUN git clone --depth 1 https://github.com/H3xKatana/autoVolatility3.git /opt/autovolatility3

WORKDIR /opt/autovolatility3

RUN chmod +x autovol3.py
ADD wrapper.sh /opt/.

# default command (you can override at runtime)
ENTRYPOINT ["bash", "/opt/wrapper.sh"]
