"""Unix-socket server that hosts the real-time engine for a front end.

Run with ``python -m engine.server``.  The socket is bound before the heavy
Torch/engine imports so a front end can connect immediately and show startup
progress; commands sent before the ``ready`` event fail with ``not_ready``.
"""

import argparse
import asyncio
import collections
import concurrent.futures
import datetime
import os
import signal
import socket
import sys
import threading
import traceback

from engine.protocol import PROTOCOL_VERSION, decode, default_socket_path, encode

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
MAX_LOG_CHARS = 40000
TICK_SECONDS = 0.05
MAX_CLIENT_BUFFER_BYTES = 4 * 1024 * 1024
# sockaddr_un.sun_path is 108 bytes including the terminating NUL.
MAX_SOCKET_PATH_BYTES = 107
#: Longest wait for the engine to stop its streams on exit.  Closing an audio
#: device can hang inside PortAudio/JACK; the settings are saved before that,
#: so past this the process exits anyway instead of lingering.
SHUTDOWN_TIMEOUT_SECONDS = 5.0


class CommandError(Exception):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class LogWriter:
    """Mirror stdout/stderr to the terminal and to connected clients."""

    def __init__(self, server, mirror):
        self.server = server
        self.mirror = mirror

    def write(self, text):
        if not text:
            return 0
        if self.mirror is not None:
            self.mirror.write(text)
        self.server.append_log(str(text))
        return len(text)

    def flush(self):
        if self.mirror is not None:
            self.mirror.flush()

    def isatty(self):
        return False


def socket_in_use(path):
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(path)
        return True
    except OSError:
        return False
    finally:
        probe.close()


class EngineServer:
    def __init__(self, socket_path, engine_factory=None, exit_when_idle=None):
        self.socket_path = socket_path
        self.engine_factory = engine_factory or self.default_engine_factory
        self.exit_when_idle = exit_when_idle
        self.engine = None
        self.init_error = None
        self.clients = set()
        self.log_chunks = collections.deque()
        self.log_size = 0
        self.log_lock = threading.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rvc-engine"
        )
        self.loop = None
        self.stopping = None
        self.tick_pending = False
        self.idle_since = None

    @staticmethod
    def default_engine_factory(emit):
        from engine.core import RealtimeEngine

        return RealtimeEngine(emit)

    # -- events -----------------------------------------------------------
    def emit(self, event, data=None):
        """Thread-safe: broadcast one event to every connected client."""
        message = encode({"event": event, "data": data})
        if self.loop is None or self.loop.is_closed():
            return
        try:
            self.loop.call_soon_threadsafe(self.broadcast, message)
        except RuntimeError:
            pass

    def broadcast(self, message):
        for writer in list(self.clients):
            self.send(writer, message)

    def send(self, writer, message):
        transport = writer.transport
        if transport.is_closing():
            self.clients.discard(writer)
            return
        if transport.get_write_buffer_size() > MAX_CLIENT_BUFFER_BYTES:
            # A client that stopped reading must not grow memory without bound.
            self.clients.discard(writer)
            writer.close()
            return
        writer.write(message)

    def append_log(self, text):
        with self.log_lock:
            self.log_chunks.append(text)
            self.log_size += len(text)
            while self.log_size > MAX_LOG_CHARS and len(self.log_chunks) > 1:
                self.log_size -= len(self.log_chunks.popleft())
        self.emit("log", {"text": text})

    def log_text(self):
        with self.log_lock:
            return "".join(self.log_chunks)[-MAX_LOG_CHARS:]

    def clear_log(self):
        with self.log_lock:
            self.log_chunks.clear()
            self.log_size = 0

    def save_log(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(LOGS_DIR, f"RuntimeLog_{timestamp}.txt")
        try:
            os.makedirs(LOGS_DIR, exist_ok=True)
            with open(path, "w", encoding="utf-8") as log_file:
                log_file.write(self.log_text())
        except OSError as error:
            raise CommandError("log_save_failed", str(error)) from error
        return {"path": path}

    # -- engine thread ----------------------------------------------------
    async def on_engine_thread(self, function, *args):
        return await self.loop.run_in_executor(self.executor, function, *args)

    def init_engine(self):
        try:
            self.engine = self.engine_factory(self.emit)
        except BaseException as error:
            self.init_error = f"{error}\n{traceback.format_exc()}"
            print(traceback.format_exc())
            self.emit("fatal", {"code": "init_failed", "message": self.init_error})
            return
        self.emit("ready", self.engine.state())

    def run_tick(self):
        try:
            self.engine.tick()
        except Exception:
            print(traceback.format_exc())
        finally:
            self.tick_pending = False

    async def tick_loop(self):
        while not self.stopping.is_set():
            await asyncio.sleep(TICK_SECONDS)
            if self.engine is not None and not self.tick_pending:
                self.tick_pending = True
                self.loop.run_in_executor(self.executor, self.run_tick)
            if self.exit_when_idle is not None:
                self.check_idle()

    def check_idle(self):
        if self.clients:
            self.idle_since = None
            return
        now = self.loop.time()
        if self.idle_since is None:
            self.idle_since = now
        elif now - self.idle_since >= self.exit_when_idle:
            print("No clients connected; shutting down.")
            self.stopping.set()

    # -- commands ---------------------------------------------------------
    def command_table(self):
        engine = self.engine

        def require(args, key, kind):
            value = args.get(key)
            if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
                raise CommandError("bad_request", f"'{key}' is required")
            return value

        return {
            "get_state": lambda args: engine.state(),
            "update_settings": lambda args: engine.update_settings(
                require(args, "settings", dict)
            ),
            "reset_settings": lambda args: engine.reset_settings(require(args, "group", str)),
            "reload_models": lambda args: engine.reload_models(),
            "delete_model": lambda args: engine.delete_model(require(args, "name", str)),
            "rename_model": lambda args: engine.rename_model(
                require(args, "name", str), require(args, "new_name", str)
            ),
            "import_model": lambda args: {"models": engine.import_model(require(args, "paths", list))},
            "reload_devices": lambda args: engine.reload_devices(),
            "start": lambda args: engine.start(args.get("function", "vc")),
            "stop": lambda args: engine.stop(),
            "file_select": lambda args: engine.file_select(require(args, "path", str)),
            "file_play": lambda args: engine.file_play(),
            "file_pause": lambda args: engine.file_pause(),
            "file_stop": lambda args: engine.file_stop(),
            "file_seek": lambda args: engine.file_seek(args.get("seconds")),
            "record_start": lambda args: engine.start_recording(),
            "record_stop": lambda args: {"paths": engine.stop_recording()},
            "analyze_source_pitch": lambda args: engine.analyze_source_pitch(
                require(args, "paths", list)
            ),
            "reset_voice_pitch": lambda args: engine.reset_voice_pitch(),
        }

    LOG_COMMANDS = ("get_log", "clear_log", "save_log", "shutdown", "ping")

    async def run_command(self, name, args):
        if name == "ping":
            return {"protocol": PROTOCOL_VERSION, "ready": self.engine is not None}
        if name == "get_log":
            return {"text": self.log_text()}
        if name == "clear_log":
            self.clear_log()
            return None
        if name == "save_log":
            return self.save_log()
        if name == "shutdown":
            self.stopping.set()
            return None
        if self.engine is None:
            if self.init_error:
                raise CommandError("init_failed", self.init_error)
            raise CommandError("not_ready", "The engine is still starting.")
        handler = self.command_table().get(name)
        if handler is None:
            raise CommandError("unknown_command", f"Unknown command: {name}")
        return await self.on_engine_thread(handler, args)

    async def handle_message(self, writer, line):
        request_id = None
        try:
            message = decode(line)
            request_id = message.get("id")
            name = message.get("cmd")
            args = message.get("args") or {}
            if not isinstance(name, str) or not isinstance(args, dict):
                raise CommandError("bad_request", "'cmd' must be a string and 'args' an object")
            result = await self.run_command(name, args)
            response = {"id": request_id, "ok": True, "result": result}
        except (CommandError, Exception) as error:
            code = getattr(error, "code", None)
            if code is None:
                if isinstance(error, ValueError):
                    code, message_text, details = "bad_request", str(error), {}
                else:
                    print(traceback.format_exc())
                    code, message_text, details = "internal_error", str(error), {}
            else:
                message_text = getattr(error, "message", str(error))
                details = getattr(error, "details", {})
            response = {
                "id": request_id,
                "ok": False,
                "error": {"code": code, "message": message_text, "details": details},
            }
        self.send(writer, encode(response))

    async def handle_client(self, reader, writer):
        self.clients.add(writer)
        state = None
        if self.engine is not None:
            state = await self.on_engine_thread(self.engine.state)
        self.send(
            writer,
            encode(
                {
                    "event": "hello",
                    "data": {
                        "protocol": PROTOCOL_VERSION,
                        "ready": self.engine is not None,
                        "init_error": self.init_error,
                        "state": state,
                    },
                }
            ),
        )
        try:
            while not self.stopping.is_set():
                try:
                    line = await reader.readline()
                except (ConnectionError, asyncio.LimitOverrunError, ValueError):
                    break
                if not line:
                    break
                if line.strip():
                    # Commands from one client are answered in order.
                    await self.handle_message(writer, line)
        finally:
            self.clients.discard(writer)
            writer.close()

    # -- lifetime ---------------------------------------------------------
    async def serve(self):
        self.loop = asyncio.get_running_loop()
        self.stopping = asyncio.Event()
        os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)
        if len(os.fsencode(self.socket_path)) > MAX_SOCKET_PATH_BYTES:
            raise SystemExit(f"Socket path is too long for AF_UNIX: {self.socket_path}")
        if os.path.exists(self.socket_path):
            if socket_in_use(self.socket_path):
                raise SystemExit(f"Another engine is already listening on {self.socket_path}")
            os.unlink(self.socket_path)
        server = await asyncio.start_unix_server(
            self.handle_client, path=self.socket_path, limit=1024 * 1024
        )
        os.chmod(self.socket_path, 0o600)
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                self.loop.add_signal_handler(signum, self.stopping.set)
        print(f"Engine listening on {self.socket_path}")
        self.loop.run_in_executor(self.executor, self.init_engine)
        ticker = asyncio.create_task(self.tick_loop())
        try:
            await self.stopping.wait()
        finally:
            ticker.cancel()
            server.close()
            if self.engine is not None:
                try:
                    await asyncio.wait_for(
                        self.on_engine_thread(self.engine.shutdown), SHUTDOWN_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    print(
                        "Engine shutdown did not finish within %.0f s "
                        "(an audio device did not close); exiting anyway."
                        % SHUTDOWN_TIMEOUT_SECONDS
                    )
                except Exception:
                    print(traceback.format_exc())
            for writer in list(self.clients):
                writer.close()
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
            self.executor.shutdown(wait=False, cancel_futures=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="RVC real-time engine server")
    parser.add_argument("--socket", default=default_socket_path(), help="Unix socket path")
    parser.add_argument(
        "--exit-when-idle",
        type=float,
        metavar="SECONDS",
        help="Exit after no client has been connected for this long",
    )
    options = parser.parse_args(argv)
    server = EngineServer(options.socket, exit_when_idle=options.exit_when_idle)
    sys.stdout = LogWriter(server, sys.stdout)
    sys.stderr = LogWriter(server, sys.stderr)
    asyncio.run(server.serve())
    print("Engine stopped.")
    sys.stdout.flush()
    sys.stderr.flush()
    # Exit now: a worker thread stuck in an audio driver, or library threads
    # (JACK, CUDA), would otherwise keep the process alive at interpreter exit.
    os._exit(0)


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ["OMP_NUM_THREADS"] = "4"
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    main()
