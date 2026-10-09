ARG IMAGE="ubuntu:22.04"
FROM $IMAGE

WORKDIR /app

RUN apt-get update && \
    apt-get install python2 git -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://bootstrap.pypa.io/pip/2.7/get-pip.py get-pip.py
RUN python2 get-pip.py && \
    pip2 install --no-cache-dir balbuzard
# the balbuzard commit whose scripts the wrapper runs (the project publishes no releases)
# renovate: datasource=git-refs depName=https://github.com/decalage2/balbuzard currentValue=master
ARG BALBUZARD_COMMIT="16c1e21a5ea0b29237746760ab72ef9683338dc5"

# fetch the pinned commit and verify it
RUN git init -q balbuzard && \
    git -C balbuzard fetch -q --depth 1 https://github.com/decalage2/balbuzard.git "$BALBUZARD_COMMIT" && \
    git -C balbuzard checkout -q --detach FETCH_HEAD && \
    test "$(git -C balbuzard rev-parse HEAD)" = "$BALBUZARD_COMMIT"

COPY wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]
