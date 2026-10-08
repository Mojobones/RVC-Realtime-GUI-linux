"""Command-line client for the engine server.

Examples:
    python -m engine.cli state
    python -m engine.cli devices
    python -m engine.cli set input_device="[JACK] Elgato Wave XLR Mono" pitch=12
    python -m engine.cli start            # or: start passthrough
    python -m engine.cli watch            # stream events until Ctrl+C
    python -m engine.cli call file_seek '{"seconds": 30}'
"""

import argparse
import json
import socket
import sys

from engine.protocol import decode, default_socket_path, encode


class EngineClient:
    def __init__(self, path, timeout=None):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(path)
        self.reader = self.sock.makefile("rb")
        self.next_id = 1
        self.hello = self.read_message()

    def close(self):
        self.reader.close()
        self.sock.close()

    def read_message(self):
        line = self.reader.readline()
        if not line:
            raise ConnectionError("engine closed the connection")
        return decode(line)

    def call(self, cmd, args=None, on_event=None):
        request_id = self.next_id
        self.next_id += 1
        self.sock.sendall(encode({"id": request_id, "cmd": cmd, "args": args or {}}))
        while True:
            message = self.read_message()
            if message.get("id") == request_id and "ok" in message:
                return message
            if on_event is not None and "event" in message:
                on_event(message)

    def events(self):
        while True:
            yield self.read_message()


def parse_assignment(text):
    key, separator, raw = text.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def print_event(message, verbose_meters=False):
    event, data = message["event"], message.get("data")
    if event == "log":
        sys.stdout.write(data["text"])
        sys.stdout.flush()
    elif event == "meters":
        if verbose_meters:
            print(f"[meters] {json.dumps(data)}")
    elif event == "state":
        print(f"[state] running={data['running']} function={data['function']}")
    else:
        print(f"[{event}] {json.dumps(data, ensure_ascii=False)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="RVC engine client")
    parser.add_argument("--socket", default=default_socket_path())
    parser.add_argument("--meters", action="store_true", help="print meter events too")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("state")
    sub.add_parser("devices")
    sub.add_parser("models")
    set_parser = sub.add_parser("set")
    set_parser.add_argument("assignments", nargs="+", type=parse_assignment)
    start_parser = sub.add_parser("start")
    start_parser.add_argument("function", nargs="?", default="vc", choices=("vc", "passthrough"))
    sub.add_parser("stop")
    sub.add_parser("log")
    sub.add_parser("watch")
    call_parser = sub.add_parser("call")
    call_parser.add_argument("cmd")
    call_parser.add_argument("args", nargs="?", default="{}", type=json.loads)
    options = parser.parse_args(argv)

    try:
        client = EngineClient(options.socket)
    except OSError as error:
        raise SystemExit(f"Cannot connect to {options.socket}: {error}")

    def show(message):
        print_event(message, options.meters)

    command = options.command
    if command == "watch":
        hello = client.hello["data"]
        print(f"[hello] protocol={hello['protocol']} ready={hello['ready']}")
        try:
            for message in client.events():
                if "event" in message:
                    show(message)
        except KeyboardInterrupt:
            return 0
    if command in ("state", "devices", "models"):
        response = client.call("get_state")
    elif command == "set":
        response = client.call("update_settings", {"settings": dict(options.assignments)})
    elif command == "start":
        response = client.call("start", {"function": options.function}, on_event=show)
    elif command == "stop":
        response = client.call("stop", on_event=show)
    elif command == "log":
        response = client.call("get_log")
    else:
        response = client.call(options.cmd, options.args, on_event=show)

    if not response["ok"]:
        error = response["error"]
        print(f"error [{error['code']}]: {error['message']}", file=sys.stderr)
        return 1
    result = response["result"]
    if command == "devices":
        result = result["devices"]
    elif command == "models":
        result = result["models"]
    elif command == "log":
        sys.stdout.write(result["text"])
        return 0
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
