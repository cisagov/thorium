ARG IMAGE="python:3.11-slim"
FROM $IMAGE

# install dependencies
RUN pip3 install pycryptodome yara-python

# install volatility3
RUN pip3 install volatility3

# make sure our symbols directory is writable by all users
RUN chmod -R 777 /usr/local/lib/python3.*/site-packages/volatility3

# add our wrapper
ADD wrapper.sh /app/.

ENTRYPOINT ["bash", "/app/wrapper.sh"]
