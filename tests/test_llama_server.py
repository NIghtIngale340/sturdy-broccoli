"""LlamaClient.complete trims a completion that ends mid-UTF-8 character
(llama-server b49650a returns HTTP 500 for it). Run: python3 tests/test_llama_server.py"""
import io, json, os, sys, urllib.error, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import llama_server as ls


def http500(req, body):
    return urllib.error.HTTPError(req.full_url, 500, "err", {}, io.BytesIO(body))


def broken_at_10(req, timeout):
    if json.loads(req.data)["n_predict"] >= 10:
        raise http500(req, b'{"error":{"message":"The model produced output that '
                           b'does not match the expected Content-only format"}}')
    return io.BytesIO(json.dumps({"content": "Business "}).encode())


def other_500(req, timeout):
    raise http500(req, b'{"error":"boom"}')


ls.time.sleep = lambda s: None

urllib.request.urlopen = broken_at_10
c = ls.LlamaClient(1)
assert c.complete("p", 10) == "Business "
assert c.utf8_trimmed == 1

urllib.request.urlopen = other_500   # any other 500 still fails, untrimmed
try:
    c.complete("p", 10)
    raise AssertionError("expected LlamaServerError")
except ls.BrokenUtf8Output:
    raise AssertionError("unrelated 500 treated as a UTF-8 cut")
except ls.LlamaServerError:
    pass
assert c.utf8_trimmed == 1
print("test_llama_server: OK")
