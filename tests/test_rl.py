import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_rl import format_reward_func

class TestRLTrainer(unittest.TestCase):
    def test_format_reward_func_exact_match(self):
        completions = [
            [{"role": "assistant", "content": "The answer is 42"}],
            [{"role": "assistant", "content": "hello world"}],
            "just a string match",
            "This is wrong"
        ]
        ground_truths = [
            "42",
            "world",
            "match",
            "correct answer"
        ]
        
        rewards = format_reward_func(completions, ground_truth=ground_truths)
        
        self.assertEqual(len(rewards), 4)
        self.assertEqual(rewards[0], 1.0) # "42" is in "the answer is 42"
        self.assertEqual(rewards[1], 1.0) # "world" is in "hello world"
        self.assertEqual(rewards[2], 1.0) # "match" is in "just a string match"
        self.assertEqual(rewards[3], 0.0) # "correct answer" not in "this is wrong"

if __name__ == '__main__':
    unittest.main()
