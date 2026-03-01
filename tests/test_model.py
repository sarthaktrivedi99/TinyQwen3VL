import unittest
import sys
import os
import torch
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PatchMerger, TinyQwen3VLConfig

class TestModelComponents(unittest.TestCase):
    def test_patch_merger_forward(self):
        # Qwen3VL PatchMerger does a 2x2 spatial merge + MLP
        vision_dim = 64
        llm_dim = 128
        merge_size = 2
        merger = PatchMerger(vision_dim=vision_dim, llm_dim=llm_dim, merge_size=merge_size)
        
        # Dummy inputs
        B = 2
        grid_h = 4
        grid_w = 4
        N = grid_h * grid_w
        dummy_x = torch.randn(B, N, vision_dim)
        
        out = merger(dummy_x, grid_h, grid_w)
        
        # Expected shape after 2x2 merge:
        expected_n = (grid_h // merge_size) * (grid_w // merge_size)
        
        self.assertEqual(out.shape, (B, expected_n, llm_dim))

    def test_config_initialization(self):
        config = TinyQwen3VLConfig(
            llm_model_id="dummy_llm",
            vision_model_id="dummy_vision",
            freeze_vision=True,
            freeze_llm=False
        )
        self.assertEqual(config.model_type, "tiny_qwen3_vl")
        self.assertEqual(config.llm_model_id, "dummy_llm")
        self.assertTrue(config.freeze_vision)
        self.assertFalse(config.freeze_llm)

if __name__ == '__main__':
    unittest.main()
