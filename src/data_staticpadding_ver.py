from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import omegaconf
from typing import Union, List, Tuple, Literal


class IMDBDatset(torch.utils.data.Dataset):
    def __init__(self, data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']):
        """
        Inputs :
                data_config : omegaconf.DictConfig{
                    model_name : str   # 'bert-base-uncased' or 'answerdotai/ModernBERT-base'
                    max_len    : int   # default 128
                    seed       : int   # default 42
                    train_frac : float # default 0.8
                    val_frac   : float # default 0.1
                    test_frac  : float # default 0.1
                    batch_size : int   # dataloader에서 사용
                    num_workers: int   # dataloader에서 사용
                    }
            split : Literal['train', 'valid', 'test']
        Outputs : None
        """

        #self.split: split
        self.max_len: int = int(getattr(data_config, "max_len", 128)) # 조정 필요
        self.seed: int = int(getattr(data_config, 'seed', 42))
        self.tokenizer: AutoTokenizer.from_pretrained(data_config.model_name, use_fast=True) 
        
        # 분할 비율
        train_frac = float(getattr(data_config, 'train_frac', 0.8))
        val_frac   = float(getattr(data_config, 'val_frac',   0.1))
        test_frac  = float(getattr(data_config, 'test_trac',  0.1))
        s = train_frac + val_frac + test_frac

        if not(abs(s-1.0) < 1e-8):
            raise ValueError(f"train+val+test fractions must sum to 1.0, got {s}")

        # imdb 로드 후 25k+25k
        ## pip install -U datasets 으로부터
        raw = load_dataset('stanfordnlp/imdb')
        merged = concatenate_datasets([raw['train'], raw['test']])

        # train vs (val+test)
        temp_frac = val_frac + test_frac
        first = merged.train_test_split(test_size=temp_frac, seed=42, stratify_by_column="label")
        train_set = first["train"]
        temp_set  = first["test"]

        rel_test = test_frac / temp_frac
        second = temp_set.train_test_split(test_size=rel_test, seed=42, stratify_by_column="label")
        val_set, test_set = second["train"], second["test"]

        if split == "train":
            chosen = train_set
        elif split == "valid":
            if valid_set is None:
                raise RuntimeError("valid split is empty—check fractions.")
            chosen = valid_set
        else:
            if test_set is None:
                raise RuntimeError("test split is empty—check fractions.")
            chosen = test_set

        self.texts: List[str] = list(chosen["text"])
        self.labels: List[int] = list(chosen["label"])

        print(f">> SPLIT : {self.split} | Total Data Length : ", len(self.data['text']))


    def __len__(self):
        return len(self.texts)
        #raise NotImplementedError

    def __getitem__(self, idx) -> Tuple[dict, int]:
        """
        Inputs :
            idx : int
        Outputs :
            inputs : dict
                - input_ids      : List[int]   # 길이 = L_i (샘플별 가변, L_i <= max_len)
                - attention_mask : List[int]   # 길이 = L_i (실토큰=1)
                - token_type_ids : List[int]   # 길이 = L_i (있을 때만; BERT 계열은 보통 0)
            label  : int
                * 클래스 인덱스(예: 0=negative, 1=positive)
        """

        """
        Inputs :
            idx : int
        Outputs :
            inputs : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor
                attention_mask : torch.Tensor
            }
            label : int
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,      # 패딩은 collate에서 처리
            return_tensors=None
        )
        return inputs, label
        #raise NotImplementedError

    @staticmethod
    def collate_fn(batch : List[Tuple[dict, int]]) -> dict:
        """
        Inputs :
            batch : List[Tuple[dict, int]]
        Outputs :
            data_dict : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor
                attention_mask : torch.Tensor
                label : torch.Tensor
            }
        """
        raise NotImplementedError
    
def get_dataloader(data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']) -> torch.utils.data.DataLoader:
    """
        주어진 split에 대한 PyTorch DataLoader를 생성
        (train split에서는 셔플이 켜지고, valid/test에서는 꺼짐)

        Inputs :
            data_config : omegaconf.DictConfig{
                batch_size : int (default=32)
                num_workers: int (default=2)
                }
                - 그 외 Dataset에 필요한 필드(model_name, max_len, seed, train/val/test_frac 등)

        Outputs :
            dataloader : torch.utils.data.DataLoader
                - 반복 시 각 step에서 아래 형태의 배치를 반환
                {
                    "input_ids"     : LongTensor (B, T_max),
                    "attention_mask": LongTensor (B, T_max),
                    "token_type_ids": LongTensor (B, T_max),
                    "label"         : LongTensor (B,)
                }
    """
    dataset = IMDBDatset(data_config, split)
    dataloader = DataLoader(
        dataset,
        batch_size=int(getattr(data_config, "batch_size", 32)),
        shuffle=(split == "train"),
        num_workers=int(getattr(data_config, "num_workers", 2)),
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )
    return dataloader