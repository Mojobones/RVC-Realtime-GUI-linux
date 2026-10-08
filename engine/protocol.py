"""Wire format shared by the engine server and its clients.

Messages are UTF-8 JSON objects, one per line, over a Unix stream socket.

    client -> engine   {"id": 1, "cmd": "start", "args": {"function": "vc"}}
    engine -> client   {"id": 1, "ok": true, "result": null}
                       {"id": 1, "ok": false, "error": {"code": ..., "message": ..., "details": {...}}}
                       {"event": "meters", "data": {...}}

See docs/engine-protocol.md for the commands and events.
"""

import json
import os

PROTOCOL_VERSION = 1
SOCKET_NAME = "rvc-realtime.sock"


def default_socket_path():
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/rvc-realtime-{os.getuid()}"
    return os.path.join(runtime_dir, SOCKET_NAME)


def encode(message):
    return (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line):
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("message must be a JSON object")
    return message
