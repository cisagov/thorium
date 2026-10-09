"""Fetch URLs with retries, for the CI scripts that call upstream APIs.

Uses only the Python standard library. A request is retried with exponential
backoff and jitter when it fails in a way that is likely to be transient:

* network errors: connection failures, resets, timeouts and truncated reads
* HTTP 429 and 5xx responses, honoring a `Retry-After` header
* GitHub's rate limit (HTTP 403 with `x-ratelimit-remaining: 0`), but only when
  the limit resets soon enough to wait for it

Other HTTP errors, such as 404, are raised immediately. Every failure is raised
as `HttpError`, whose `status` is the HTTP status, or None for network errors.

Import it from a sibling script as `import http_retry`; scripts run as
`python3 .github/scripts/<script>.py` have this directory on `sys.path`.
"""

import email.utils
import http.client
import json
import random
import socket
import time
import urllib.error
import urllib.request

# the longest a single wait may be, so a hostile or broken server cannot stall a job
MAX_WAIT_SECONDS = 60
# the base of the exponential backoff between attempts
BACKOFF_SECONDS = 2


class HttpError(Exception):
    """A request that failed, after any retries.

    `status` is the HTTP status code, or None when no response was received.
    """

    def __init__(self, url, message, status=None):
        super().__init__(f"{url}: {message}")
        self.url = url
        self.status = status


def _retry_after(headers):
    """Read a `Retry-After` header as a number of seconds.

    # Arguments

    * `headers` - The response headers

    Returns the delay in seconds, or None when the header is missing or invalid.
    """
    value = (headers.get("Retry-After") or "").strip() if headers else ""
    if not value:
        return None
    if value.isdigit():
        return float(value)
    # the header may also be an HTTP date
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def _rate_limit_wait(error):
    """Decide how long to wait for a GitHub rate limit to reset.

    # Arguments

    * `error` - The HTTP 403 error

    Returns the delay in seconds, or None when the 403 is not a rate limit or the
    limit resets too far in the future to wait for.
    """
    headers = error.headers
    if not headers or headers.get("x-ratelimit-remaining") != "0":
        return None
    try:
        wait = float(headers.get("x-ratelimit-reset", "")) - time.time()
    except ValueError:
        return None
    return max(0.0, wait) + 1 if wait <= MAX_WAIT_SECONDS else None


def get_bytes(url, headers=None, timeout=30, attempts=4):
    """Fetch a URL and return its body.

    # Arguments

    * `url` - The URL to fetch
    * `headers` - Optional request headers
    * `timeout` - The socket timeout of each attempt, in seconds
    * `attempts` - How many times to try before giving up

    Raises `HttpError` when every attempt failed or the error is not retryable.
    """
    attempts = max(1, attempts)
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers=headers or {})
        wait = None
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            status = error.code
            if status == 429 or status >= 500:
                wait = _retry_after(error.headers)
            elif status == 403:
                wait = _rate_limit_wait(error)
                if wait is None:
                    raise HttpError(url, f"HTTP {status} {error.reason}", status) from error
            else:
                raise HttpError(url, f"HTTP {status} {error.reason}", status) from error
            failure = HttpError(url, f"HTTP {status} {error.reason}", status)
        # URLError wraps connection failures; the rest can surface while reading the body
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, http.client.HTTPException) as error:
            reason = getattr(error, "reason", None) or error
            failure = HttpError(url, f"request failed: {reason}")
        if attempt == attempts:
            raise failure
        # exponential backoff with jitter, unless the server asked for a specific delay
        if wait is None:
            wait = BACKOFF_SECONDS * 2 ** (attempt - 1) + random.uniform(0, 1)
        time.sleep(min(wait, MAX_WAIT_SECONDS))
    raise AssertionError("unreachable")


def get_json(url, headers=None, timeout=30, attempts=4):
    """Fetch a URL and decode its body as JSON.

    # Arguments

    * `url` - The URL to fetch
    * `headers` - Optional request headers
    * `timeout` - The socket timeout of each attempt, in seconds
    * `attempts` - How many times to try before giving up

    Raises `HttpError` when the request failed or the body is not valid JSON.
    """
    body = get_bytes(url, headers=headers, timeout=timeout, attempts=attempts)
    try:
        return json.loads(body)
    except ValueError as error:
        raise HttpError(url, f"response is not valid JSON: {error}") from error
