"""GPU work from another thread must not break CUDA Graph captures.

The pitch analysis runs its own RMVPE on a worker thread while the audio
thread converts.  If the worker captured graphs, or ran kernels while the
audio thread captured one, the capture failed ("operation not permitted
when stream is capturing"), conversion went silent and the allocator was
left corrupted until the engine restarted.
"""

import threading
import time
import unittest

import torch

from tools import cuda_graph


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class SideGpuWorkTest(unittest.TestCase):
    def setUp(self):
        previous = cuda_graph.os.environ.get(cuda_graph.ENV_NAME)
        cuda_graph.os.environ[cuda_graph.ENV_NAME] = "1"

        def restore():
            if previous is None:
                cuda_graph.os.environ.pop(cuda_graph.ENV_NAME, None)
            else:
                cuda_graph.os.environ[cuda_graph.ENV_NAME] = previous

        self.addCleanup(restore)
        self.device = torch.device("cuda", torch.cuda.device_count() - 1)
        self.layer = torch.nn.Linear(64, 64).to(self.device)

    def test_side_work_never_captures(self):
        owner = torch.nn.Module()
        inputs = torch.randn(4, 64, device=self.device)
        with cuda_graph.side_gpu_work():
            result = cuda_graph.run_cuda_graph(owner, "side", self.layer, inputs)
        torch.testing.assert_close(result, self.layer(inputs))
        self.assertEqual(cuda_graph.get_cuda_graph_stats(owner)["captures"], 0)

    def test_captures_wait_for_side_work(self):
        entered, release = threading.Event(), threading.Event()

        def side():
            with cuda_graph.side_gpu_work():
                entered.set()
                release.wait(5)

        worker = threading.Thread(target=side)
        worker.start()
        entered.wait(5)
        timer = threading.Timer(0.3, release.set)
        timer.start()
        started = time.perf_counter()
        cuda_graph.run_cuda_graph(
            torch.nn.Module(), "capture", self.layer, torch.randn(4, 64, device=self.device)
        )
        waited = time.perf_counter() - started
        worker.join()
        self.assertGreaterEqual(waited, 0.25)

    def test_captures_survive_concurrent_side_work(self):
        stop, errors = threading.Event(), []
        weights = torch.randn(512, 512, device=self.device)

        def side():
            # Allocations and kernels, as the analysis thread does.
            while not stop.is_set():
                try:
                    with cuda_graph.side_gpu_work():
                        (weights @ torch.randn(512, 512, device=self.device)).sum().item()
                except Exception as error:  # noqa: BLE001 - reported below
                    errors.append(error)
                    return

        worker = threading.Thread(target=side)
        worker.start()
        try:
            owner = torch.nn.Module()
            # A new shape each time forces a fresh capture.
            for rows in range(1, 40):
                inputs = torch.randn(rows, 64, device=self.device)
                result = cuda_graph.run_cuda_graph(owner, "audio", self.layer, inputs)
                torch.testing.assert_close(result, self.layer(inputs))
        finally:
            stop.set()
            worker.join()
        self.assertEqual(errors, [])
        stats = cuda_graph.get_cuda_graph_stats(owner)
        self.assertEqual(stats["failures"], 0)
        self.assertEqual(stats["captures"], 39)
        torch.cuda.empty_cache()  # the allocator is still healthy


if __name__ == "__main__":
    unittest.main()
