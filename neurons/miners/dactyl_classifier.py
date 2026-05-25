import os

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from miners.deberta_classifier import SimpleTestDataset


def _generate_dactyl_predictions(model, tokenizer, test_dataset, device):
    """DACTYL uses a single logit + sigmoid (EXM training), not 2-class softmax."""
    data_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    all_predictions = []
    with torch.no_grad():
        for batch in data_loader:
            token_sequences = batch.input_ids.to(device)
            attention_masks = batch.attention_mask.to(device)
            with torch.cuda.amp.autocast():
                logits = model(token_sequences, attention_masks).logits
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits.squeeze(-1))
            else:
                probs = logits.softmax(dim=1)[:, 1]
            all_predictions.append(probs.cpu().numpy())
    return np.concatenate(all_predictions)


class DactylClassifier:
    """DACTYL EXM detector (microsoft/deberta-v3-large), loaded from a HF model folder."""

    def __init__(self, model_path, device, max_length=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = device
        self.model = self.model.to(device).eval()

        if max_length is not None:
            self.max_length = max_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.max_length = self.model.config.max_position_embeddings
        else:
            self.max_length = 512

    def predict_batch(self, texts):
        test_dataset = SimpleTestDataset(texts, self.tokenizer, self.max_length)
        return _generate_dactyl_predictions(
            self.model, self.tokenizer, test_dataset, self.device
        )

    def __call__(self, text):
        return self.predict_batch([text])[0]
