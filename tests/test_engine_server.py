import asyncio
import os
import tempfile
import threading
import time
import unittest

from engine.cli import EngineClient
from engine.server import CommandError, EngineServer


class FakeEngine:
    """Stands in for RealtimeEngine so the protocol tests need no Torch."""

    def __init__(self, emit, ready_event=None):
        if ready_event is not None:
            ready_event.wait(5)
        self.emit = emit
        self.settings = {"pitch": 0.0}
        self.ticks = 0
        self.shut_down = False

    def state(self):
        return {"running": False, "function": "vc", "settings": dict(self.settings)}

    def update_settings(self, changes):
        if "pitch" not in changes:
            raise CommandError("bad_setting", "only pitch is supported", key="x")
        self.settings["pitch"] = changes["pitch"]
        self.emit("state", self.state())

    def start(self, function):
        print(f"starting {function}")
        raise RuntimeError("no audio devices in tests")

    def tick(self):
        self.ticks += 1
        if self.ticks == 1:
            self.emit("meters", {"input": 0.5})

    def shutdown(self):
        self.shut_down = True


class HangingShutdownEngine(FakeEngine):
    """Shutdown never returns, like an audio device that will not close."""

    def shutdown(self):
        self.shut_down = True
        threading.Event().wait(2)


class EngineServerTest(unittest.TestCase):
    def start_server(self, engine_factory, exit_when_idle=None):
        self.directory = tempfile.mkdtemp(prefix="rvc-")
        self.socket_path = os.path.join(self.directory, "e.sock")
        self.server = EngineServer(
            self.socket_path, engine_factory=engine_factory, exit_when_idle=exit_when_idle
        )
        self.thread = threading.Thread(target=asyncio.run, args=(self.server.serve(),))
        self.thread.start()
        for _ in range(200):
            if os.path.exists(self.socket_path):
                return
            time.sleep(0.01)
        self.fail("server did not bind its socket")

    def stop_server(self):
        if self.thread.is_alive():
            client = EngineClient(self.socket_path, timeout=5)
            client.call("shutdown")
            client.close()
        self.thread.join(5)
        self.assertFalse(self.thread.is_alive())
        self.assertFalse(os.path.exists(self.socket_path))
        os.rmdir(self.directory)

    def connect_ready(self):
        client = EngineClient(self.socket_path, timeout=5)
        self.addCleanup(client.close)
        if not client.hello["data"]["ready"]:
            while client.read_message().get("event") != "ready":
                pass
        return client

    def test_socket_accepts_commands_before_engine_is_ready(self):
        release = threading.Event()
        self.start_server(lambda emit: FakeEngine(emit, release))
        client = EngineClient(self.socket_path, timeout=5)

        self.assertFalse(client.hello["data"]["ready"])
        self.assertTrue(client.call("ping")["result"]["protocol"] >= 1)
        self.assertEqual(client.call("get_state")["error"]["code"], "not_ready")
        release.set()
        while client.read_message().get("event") != "ready":
            pass
        self.assertTrue(client.call("get_state")["ok"])
        client.close()
        self.stop_server()

    def test_commands_results_errors_and_events(self):
        self.start_server(FakeEngine)
        client = self.connect_ready()
        events = []

        response = client.call(
            "update_settings", {"settings": {"pitch": 3}}, on_event=events.append
        )
        self.assertTrue(response["ok"])
        while not any(event["event"] == "state" for event in events):
            events.append(client.read_message())
        self.assertEqual(
            client.call("get_state")["result"]["settings"]["pitch"], 3
        )

        error = client.call("update_settings", {"settings": {"x": 1}})["error"]
        self.assertEqual((error["code"], error["details"]), ("bad_setting", {"key": "x"}))
        self.assertEqual(client.call("update_settings")["error"]["code"], "bad_request")
        self.assertEqual(client.call("nope")["error"]["code"], "unknown_command")
        self.assertEqual(client.call("start")["error"]["code"], "internal_error")

        log = client.call("get_log")["result"]["text"]
        self.assertIsInstance(log, str)
        self.stop_server()
        self.assertTrue(self.server.engine.shut_down)

    def test_malformed_lines_get_an_error_response(self):
        self.start_server(FakeEngine)
        client = self.connect_ready()

        client.sock.sendall(b"not json\n")
        message = client.read_message()
        while "ok" not in message:
            message = client.read_message()

        self.assertEqual(message["error"]["code"], "bad_request")
        self.stop_server()

    def test_init_failure_is_reported(self):
        def broken(emit):
            raise RuntimeError("CUDA exploded")

        self.start_server(broken)
        client = EngineClient(self.socket_path, timeout=5)
        message = client.hello
        while message.get("event") not in ("fatal",) and not message["data"].get("init_error"):
            message = client.read_message()
        response = client.call("get_state")
        self.assertEqual(response["error"]["code"], "init_failed")
        self.assertIn("CUDA exploded", response["error"]["message"])
        client.close()
        self.stop_server()

    def test_exits_when_idle(self):
        self.start_server(FakeEngine, exit_when_idle=0.2)
        client = self.connect_ready()
        client.close()

        self.thread.join(5)
        self.assertFalse(self.thread.is_alive())
        self.assertFalse(os.path.exists(self.socket_path))
        os.rmdir(self.directory)

    def test_exits_even_if_the_engine_shutdown_hangs(self):
        from unittest import mock

        from engine import server as server_module

        with mock.patch.object(server_module, "SHUTDOWN_TIMEOUT_SECONDS", 0.3):
            self.start_server(HangingShutdownEngine, exit_when_idle=0.2)
            client = self.connect_ready()
            client.close()
            started = time.monotonic()
            self.thread.join(5)
        self.assertFalse(self.thread.is_alive())
        self.assertLess(time.monotonic() - started, 3)
        self.assertTrue(self.server.engine.shut_down)
        self.assertFalse(os.path.exists(self.socket_path))
        os.rmdir(self.directory)


if __name__ == "__main__":
    unittest.main()
