import unittest

from infer.rtrvc import RMVPE_WINDOW_STEP, rmvpe_window


def valid(window):
    return (window + 160) % RMVPE_WINDOW_STEP == 0


class RmvpeWindowTest(unittest.TestCase):
    def test_gives_at_least_the_configured_context(self):
        # 0.25 s blocks at 16 kHz with the default 2.5 s extra context.
        window = rmvpe_window(4000, available=44800)
        self.assertTrue(valid(window))
        self.assertGreaterEqual(window, int(1.28 * 16000) - 160)
        self.assertLess(window, int(1.28 * 16000) + RMVPE_WINDOW_STEP)

    def test_long_blocks_still_cover_the_block(self):
        window = rmvpe_window(24000, available=80000)  # 1.5 s blocks
        self.assertTrue(valid(window))
        self.assertGreaterEqual(window, 24000 + 800)

    def test_short_context_is_capped_to_what_is_available(self):
        # A tiny extra time leaves less than 1.28 s of 16 kHz audio.
        window = rmvpe_window(1600, available=12000)
        self.assertTrue(valid(window))
        self.assertLessEqual(window, 12000)
        self.assertGreaterEqual(window, 1600 + 800)


if __name__ == "__main__":
    unittest.main()
