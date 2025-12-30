import torch
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets


# ---------------------------------------------------------------------------
# Dataset registry: maps short name → (hf_id, config/subset, column mapping)
#
#   image_col  : column name for the PIL image
#   question_col: column name for the question string
#   answer_col : column name for the answer (str or list[str])
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    # --- FineVision multi-task (original) ---
    "finevision": {
        "hf_id": "HuggingFaceM4/FineVision",
        "format": "finevision",           # uses VQADataset
    },
    # --- Standard OCR / VQA benchmarks ---
    "textvqa": {
        "hf_id": "facebook/textvqa",
        "format": "generic",
        "image_col": "image",
        "question_col": "question",
        "answer_col": "answers",           # list[str]
    },
    "docvqa": {
        "hf_id": "lmms-lab/DocVQA",
        "format": "generic",
        "image_col": "image",
        "question_col": "query",
        "answer_col": "answers",           # list[str]
    },
    "ocrvqa": {
        "hf_id": "howard-hou/OCR-VQA",
        "format": "generic",
        "image_col": "image",
        "question_col": "question",
        "answer_col": "answers",           # list[str]
    },
    "stvqa": {
        "hf_id": "vikhyatk/stvqa",
        "format": "generic",
        "image_col": "image",
        "question_col": "question",
        "answer_col": "answers",           # list[str]
    },
    "infovqa": {
        "hf_id": "lmms-lab/InfoVQA",
        "format": "generic",
        "image_col": "image",
        "question_col": "question",
        "answer_col": "answers",           # list[str]
    },
    "chartqa": {
        "hf_id": "HuggingFaceM4/ChartQA",
        "format": "generic",
        "image_col": "image",
        "question_col": "query",
        "answer_col": "label",             # str
    },
}


# ---------------------------------------------------------------------------
# FineVision-format dataset (multi-turn, images list)
# ---------------------------------------------------------------------------
class FineVisionDataset(Dataset):
    """Dataset for HuggingFaceM4/FineVision subsets (images + texts columns)."""

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]

            # --- 1. Load Image ---
            image = None
            if 'images' in item and item['images']:
                image = item['images'][0]

            if image is None:
                print(f"[FineVision] Skipping idx {idx}: no image found")
                return None

            if image.mode != "RGB":
                image = image.convert("RGB")

            # --- 2. Prepare Text ---
            user_text = ""
            assistant_text = ""

            if 'texts' in item and isinstance(item['texts'], list):
                for turn in item['texts']:
                    if 'user' in turn:
                        user_text += turn['user']
                    if 'assistant' in turn:
                        assistant_text += turn['assistant']
            elif 'user' in item:
                user_text = item.get('user', "")
                assistant_text = item.get('assistant', "")

            return self._build_sample(image, user_text, assistant_text)

        except Exception as e:
            print(f"[FineVision] Error at index {idx}: {e}")
            return None

    def _build_sample(self, image, user_text, assistant_text):
        """Two-pass processing: prompt-only → full conversation for label masking."""

        prompt_msgs = [
            {"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": user_text}]}
        ]
        prompt_processed = self.processor.process(
            images=image, text=prompt_msgs, return_tensors="pt"
        )
        prompt_len = prompt_processed["input_ids"].shape[1]

        full_msgs = [
            {"role": "user",
             "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            {"role": "assistant",
             "content": [{"type": "text", "text": assistant_text}]}
        ]
        full_processed = self.processor.process(
            images=image, text=full_msgs, return_tensors="pt"
        )

        input_ids = full_processed["input_ids"].squeeze(0)
        labels = input_ids.clone()

        if prompt_len < labels.shape[0]:
            labels[:prompt_len] = -100
        else:
            labels[:] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[input_ids == pad_id] = -100

        return {
            "input_ids": input_ids,
            "pixel_values": full_processed["pixel_values"].squeeze(0),
            "attention_mask": full_processed["attention_mask"].squeeze(0),
            "labels": labels,
            "image_token_id": full_processed["image_token_id"],
        }


# ---------------------------------------------------------------------------
# Generic VQA dataset (image / question / answer columns)
# Works with TextVQA, DocVQA, OCR-VQA, ST-VQA, InfoVQA, ChartQA, etc.
# ---------------------------------------------------------------------------
class GenericVQADataset(Dataset):
    """
    Handles any HuggingFace VQA dataset that has image, question, and answer
    columns (the standard format for most OCR/VQA benchmarks).
    """

    def __init__(self, dataset, processor, *,
                 image_col="image", question_col="question", answer_col="answers"):
        self.dataset = dataset
        self.processor = processor
        self.image_col = image_col
        self.question_col = question_col
        self.answer_col = answer_col

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]

            # --- 1. Image ---
            image = item.get(self.image_col)
            if image is None:
                print(f"[GenericVQA] Skipping idx {idx}: no image in '{self.image_col}'")
                return None

            if image.mode != "RGB":
                image = image.convert("RGB")

            # --- 2. Question & Answer ---
            question = str(item.get(self.question_col, ""))

            raw_answer = item.get(self.answer_col, "")
            if isinstance(raw_answer, list):
                # Most benchmarks provide a list of valid answers; pick the first
                answer = str(raw_answer[0]) if raw_answer else ""
            else:
                answer = str(raw_answer)

            # --- 3. Two-pass label masking (same logic as FineVision) ---
            prompt_msgs = [
                {"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": question}]}
            ]
            prompt_len = self.processor.process(
                images=image, text=prompt_msgs, return_tensors="pt"
            )["input_ids"].shape[1]

            full_msgs = [
                {"role": "user",
                 "content": [{"type": "image"}, {"type": "text", "text": question}]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": answer}]}
            ]
            full_processed = self.processor.process(
                images=image, text=full_msgs, return_tensors="pt"
            )

            input_ids = full_processed["input_ids"].squeeze(0)
            labels = input_ids.clone()

            if prompt_len < labels.shape[0]:
                labels[:prompt_len] = -100
            else:
                labels[:] = -100

            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[input_ids == pad_id] = -100

            return {
                "input_ids": input_ids,
                "pixel_values": full_processed["pixel_values"].squeeze(0),
                "attention_mask": full_processed["attention_mask"].squeeze(0),
                "labels": labels,
                "image_token_id": full_processed["image_token_id"],
            }

        except Exception as e:
            print(f"[GenericVQA] Error at index {idx}: {e}")
            return None


# Keep backwards-compatible alias
VQADataset = FineVisionDataset


# ---------------------------------------------------------------------------
# Factory: load a dataset by short name and return the correct Dataset wrapper
# ---------------------------------------------------------------------------
def load_train_dataset(name, processor, *, subset=None, split="train",
                       max_samples=None):
    """
    Load a training dataset by short name (see DATASET_REGISTRY).

    Args:
        name:        Registry key, e.g. "textvqa", "docvqa", "finevision"
        processor:   TinyQwen3Processor instance
        subset:      Optional subset/config for FineVision (e.g. "chartqa")
        split:       HuggingFace split string; supports slicing like "train[:1000]"
        max_samples: If set, limits to first N samples via split slicing

    Returns:
        A torch Dataset ready for DataLoader
    """
    entry = DATASET_REGISTRY.get(name)
    if entry is None:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {available}"
        )

    hf_id = entry["hf_id"]
    fmt = entry["format"]

    # Build split string
    if max_samples is not None:
        split = f"{split}[:{max_samples}]"

    print(f"      Loading {name} ({hf_id}, split={split})...")

    if fmt == "finevision":
        # FineVision: requires a subset name
        if subset is None:
            subset = "chartqa"
        ds = load_dataset(hf_id, subset, split=split, trust_remote_code=True)
        return FineVisionDataset(ds, processor)
    else:
        # Generic VQA format
        config = subset  # some HF datasets use config names
        ds = load_dataset(hf_id, config, split=split, trust_remote_code=True)
        return GenericVQADataset(
            ds, processor,
            image_col=entry.get("image_col", "image"),
            question_col=entry.get("question_col", "question"),
            answer_col=entry.get("answer_col", "answers"),
        )


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return {}

    input_ids = [x['input_ids'] for x in batch]
    labels = [x['labels'] for x in batch]
    pixel_values = [x['pixel_values'] for x in batch]

    # 1. Pad Text
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100)
    attention_mask_padded = (input_ids_padded != 0).long()

    # 2. Pad Images & Create Patch Mask
    max_h = max(img.shape[1] for img in pixel_values)
    max_w = max(img.shape[2] for img in pixel_values)

    padded_images = []
    patch_masks = []
    patch_size = 16

    for img in pixel_values:
        c, h, w = img.shape
        padded_images.append(F.pad(img, (0, max_w - w, 0, max_h - h), value=0))

        # Patch-level validity mask
        n_h = h // patch_size
        n_w = w // patch_size
        grid_h = max_h // patch_size
        grid_w = max_w // patch_size

        mask = torch.zeros((grid_h, grid_w), dtype=torch.bool)
        mask[:n_h, :n_w] = True
        patch_masks.append(mask.flatten())

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
        "pixel_values": torch.stack(padded_images),
        "patch_attention_mask": torch.stack(patch_masks),
        "image_token_id": batch[0]["image_token_id"]
    }