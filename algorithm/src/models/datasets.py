from __future__ import annotations
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PositivePairDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer_name: str,
        query_text_col: str,
        candidate_text_col: str,
        max_length: int = 128,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.query_text_col = query_text_col
        self.candidate_text_col = candidate_text_col
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def _encode(self, text: str) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        q = str(row[self.query_text_col])
        c = str(row[self.candidate_text_col])

        q_enc = self._encode(q)
        c_enc = self._encode(c)

        return {
            "query_input_ids": q_enc["input_ids"],
            "query_attention_mask": q_enc["attention_mask"],
            "candidate_input_ids": c_enc["input_ids"],
            "candidate_attention_mask": c_enc["attention_mask"],
        }


class TripletTextDataset(Dataset):
    def __init__(
        self,
        triplets_df: pd.DataFrame,
        tokenizer_name: str,
        max_length: int = 128,
    ) -> None:
        self.df = triplets_df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def _encode(self, text: str) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            str(text),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        q_enc = self._encode(row["query_text"])
        p_enc = self._encode(row["positive_text"])
        n_enc = self._encode(row["negative_text"])

        return {
            "query_input_ids": q_enc["input_ids"],
            "query_attention_mask": q_enc["attention_mask"],
            "positive_input_ids": p_enc["input_ids"],
            "positive_attention_mask": p_enc["attention_mask"],
            "negative_input_ids": n_enc["input_ids"],
            "negative_attention_mask": n_enc["attention_mask"],
        }