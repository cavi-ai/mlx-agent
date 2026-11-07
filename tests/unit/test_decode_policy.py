import math
import unittest

from mlx_agent._decode_policy import LookupPolicy


class LookupPolicyTests(unittest.TestCase):
    def calibrated(self):
        policy = LookupPolicy()
        self.assertEqual(policy.proposal_limit, 0)
        policy.observe(0, 0, .01)
        self.assertEqual(policy.proposal_limit, 0)
        policy.observe(0, 0, .001)
        self.assertEqual(policy.proposal_limit, 4)
        return policy

    def test_verified_acceptance_grows_and_shrinks_the_bounded_window(self):
        policy = self.calibrated()
        policy.observe(4, 4, .001)
        self.assertEqual(policy.proposal_limit, 6)
        policy.observe(6, 6, .001)
        policy.observe(8, 8, .001)
        self.assertEqual(policy.proposal_limit, 8)
        policy.observe(8, 2, .001)
        self.assertEqual(policy.proposal_limit, 3)
        self.assertIsNone(policy.native_reason)

    def test_two_consecutive_misses_stop_speculation_for_this_request(self):
        policy = self.calibrated()
        policy.observe(4, 0, .001)
        self.assertIsNone(policy.native_reason)
        self.assertEqual(policy.proposal_limit, 2)
        policy.observe(2, 0, .001)
        self.assertEqual(policy.native_reason, "low_acceptance")
        self.assertEqual(policy.proposal_limit, 0)
        policy.observe(8, 8, .00001)
        self.assertEqual(policy.native_reason, "low_acceptance")
        self.assertIsNone(LookupPolicy().native_reason)

    def test_expensive_verified_blocks_fall_back_even_with_full_acceptance(self):
        policy = self.calibrated()
        for _ in range(3):
            policy.observe(4, 4, .01)
            self.assertIsNone(policy.native_reason)
        policy.observe(4, 4, .01)
        self.assertEqual(policy.native_reason, "verification_cost")

    def test_one_large_cost_overrun_stops_before_a_short_response_finishes(self):
        policy = self.calibrated()
        policy.observe(4, 4, .011)
        self.assertEqual(policy.native_reason, "verification_cost")
        self.assertEqual(len(policy.samples), 1)

    def test_inexpensive_verified_blocks_keep_only_four_observations(self):
        policy = self.calibrated()
        for _ in range(100):
            policy.observe(8, 8, .001)
        self.assertIsNone(policy.native_reason)
        self.assertEqual(len(policy.samples), 4)
        self.assertEqual(len(policy.calibration), 2)

    def test_invalid_timing_or_counts_select_native_instead_of_manufacturing_benefit(self):
        for proposed, accepted, elapsed in ((0, 0, 0), (0, 0, -1), (0, 0, math.nan),
                (0, 0, math.inf), (4, 5, .1), (9, 1, .1), (4, -1, .1), (True, 0, .1)):
            with self.subTest(proposed=proposed, accepted=accepted, elapsed=elapsed):
                policy = LookupPolicy()
                policy.observe(proposed, accepted, elapsed)
                self.assertEqual(policy.native_reason, "invalid_observation")

    def test_a_hit_resets_consecutive_misses(self):
        policy = self.calibrated()
        policy.observe(4, 0, .001)
        policy.observe(2, 1, .001)
        policy.observe(2, 0, .001)
        self.assertIsNone(policy.native_reason)
