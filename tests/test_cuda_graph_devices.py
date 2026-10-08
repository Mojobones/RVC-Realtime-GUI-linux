"""CUDA Graphs must work on whichever GPU the user selects.

torch.cuda.graph() defaults to one process-wide capture stream, created on the
device that was current at first use: the startup probe's GPU.  Capturing work
for another GPU on that stream fails ("operation not permitted when stream is
capturing") and poisons every later capture.
"""

import unittest

import torch

from tools import cuda_graph


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2, "needs two CUDA GPUs"
)
class SecondGpuCaptureTest(unittest.TestCase):
    def test_capture_on_a_gpu_other_than_the_probed_one(self):
        self.assertTrue(cuda_graph.detect_cuda_graph_support(torch.device("cuda:0")))
        # The default capture stream now exists on cuda:0, as after startup.
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            pass

        device = torch.device("cuda:1")
        layer = torch.nn.Linear(256, 256).to(device)
        owner = torch.nn.Module()
        previous = cuda_graph.os.environ.get(cuda_graph.ENV_NAME)
        cuda_graph.os.environ[cuda_graph.ENV_NAME] = "1"
        try:
            torch.cuda.set_device(device)  # as Config.select_cuda_device does
            for _ in range(3):
                # New values each call: a capture that silently recorded
                # nothing would keep returning the first call's output.
                inputs = torch.randn(8, 256, device=device)
                expected = layer(inputs)
                result = cuda_graph.run_cuda_graph(owner, "linear", layer, inputs)
                torch.testing.assert_close(result, expected)
        finally:
            torch.cuda.set_device(torch.device("cuda:0"))
            if previous is None:
                cuda_graph.os.environ.pop(cuda_graph.ENV_NAME, None)
            else:
                cuda_graph.os.environ[cuda_graph.ENV_NAME] = previous

        stats = cuda_graph.get_cuda_graph_stats(owner)
        self.assertEqual(stats["failures"], 0)
        self.assertEqual(stats["replays"], 3)


if __name__ == "__main__":
    unittest.main()
