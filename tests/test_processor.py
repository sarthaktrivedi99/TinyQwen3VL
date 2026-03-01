import unittest
import sys
import os
import torch
from unittest.mock import patch, MagicMock
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import TinyQwen3Processor

class TestTinyQwen3Processor(unittest.TestCase):
    @patch("src.processor.AutoTokenizer.from_pretrained")
    def test_processor_initialization_and_special_tokens(self, mock_from_pretrained):
        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_tokenizer.add_special_tokens.return_value = 3
        mock_tokenizer.convert_tokens_to_ids.return_value = 123456
        mock_from_pretrained.return_value = mock_tokenizer

        # Init processor
        processor = TinyQwen3Processor(
            vision_model_id="dummy_vision",
            llm_model_id="dummy_llm"
        )

        # Validate token settings
        self.assertEqual(processor.tokenizer.pad_token, "<|endoftext|>")
        mock_tokenizer.add_special_tokens.assert_called_once()
        self.assertEqual(processor.image_token_id, 123456)
        
    @patch("src.processor.AutoTokenizer.from_pretrained")    
    def test_processor_call_alias(self, mock_from_pretrained):
        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<|endoftext|>"
        mock_from_pretrained.return_value = mock_tokenizer
        
        processor = TinyQwen3Processor(
            vision_model_id="dummy_vision",
            llm_model_id="dummy_llm",
            patch_size=16,
            spatial_merge_size=2
        )
        
        # Mock process to just return args
        processor.process = MagicMock(return_value="processed_data")
        
        # Test __call__
        res = processor(images="dummy_image", text="dummy_text")
        self.assertEqual(res, "processed_data")
        processor.process.assert_called_with(images="dummy_image", text="dummy_text")

if __name__ == '__main__':
    unittest.main()
