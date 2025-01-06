from transformers import (
    Blip2ForConditionalGeneration,
    AutoImageProcessor, 
    AutoModel,
    AutoModelForCausalLM,
    Seq2SeqTrainer,
    EarlyStoppingCallback, 
    Seq2SeqTrainingArguments,
    AutoTokenizer,
                            )
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import pandas as pd
from accelerate import Accelerator
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import List, Dict, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

accelerate = Accelerator()

encoder_path = '/project/lt200203-aimedi/pud/gen-x-report/model/rad-dino-12c'
deocder_path = '/project/lt200203-aimedi/pud/gen-x-report/model/RadLLaMA-7b'

model = Blip2ForConditionalGeneration.from_pretrained("/project/lt200203-aimedi/pud/gen-x-report/model/blip2-opt-2.7b",
                                                    ignore_mismatched_sizes=True)

processor = AutoImageProcessor.from_pretrained(encoder_path)
encoder = AutoModel.from_pretrained(encoder_path,
                                    ignore_mismatched_sizes=True
                                    )
decoder = AutoModelForCausalLM.from_pretrained(deocder_path)
tokenizer = AutoTokenizer.from_pretrained(deocder_path,trust_remote_code=True)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["[SEP]"],
})
tokenizer.pad_token = '[PAD]'

model.language_model = decoder
model.vision_model = encoder

model.config.text_config = decoder.config
model.config.vision_config = encoder.config

model.generation_config = decoder.generation_config

model.config.hidden_size = 4096

model.language_projection = nn.Linear(in_features=768, out_features=decoder.config.hidden_size, bias=True)

model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.language_model.resize_token_embeddings(len(tokenizer))

params = model.state_dict()
embeddings = params['language_model.model.embed_tokens.weight']
pre_expansion_embeddings = embeddings[:-65,:]
mu = torch.mean(pre_expansion_embeddings, dim=0)
n = pre_expansion_embeddings.size()[0]
sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
dist = torch.distributions.multivariate_normal.MultivariateNormal(
        mu, covariance_matrix=1e-5*sigma)

new_embeddings = torch.stack(tuple((dist.sample() for _ in range(65))), dim=0)
embeddings[-65:,:] = new_embeddings
params['language_model.model.embed_tokens.weight'][-65:,:] = new_embeddings
model.load_state_dict(params)

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, tokenizer, processor, max_length=512):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def _safe_open_image(self, path):
        if pd.isna(path) or not isinstance(path, str):
            return None
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error opening image: {path}, {e}")
            return None

    def _process_image(self, image):
        if image is None:
            return None
        return self.processor(images=image, return_tensors="pt", do_rescale=False)['pixel_values'].squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        processed_views = []
        for i in range(1, 4):
            view = self._safe_open_image(row.get(f'view{i}'))
            if view is not None:
                processed_view = self._process_image(view)
                if processed_view is not None:
                    processed_views.append(processed_view)

        if not processed_views:
            print(f"Skipping index {idx} due to no valid images.")
            return self.__getitem__((idx + 1) % len(self.df))

        text = f"{row['findings']} [SEP] {row['impression']}"
        encodings = self.tokenizer(text)
        encodings.input_ids.append(self.tokenizer.eos_token_id)
        encodings.attention_mask.append(1)

        return {
            'pixel_values': processed_views,
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': encodings['input_ids'].copy()
        }

@dataclass
class DataCollatorForMultipleViews:
    tokenizer: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        
        pixel_values_list = [item.pop('pixel_values') for item in features]
        batch_text = self.tokenizer.pad(
            [{'input_ids': f['input_ids'], 
                'attention_mask': f['attention_mask']} for f in features],
            padding=True,
            return_tensors="pt"
        )
        
        labels = self.tokenizer.pad(
            [{'input_ids': f['labels']} for f in features],
            padding=True,
            return_tensors="pt"
        )
        
        max_views = max(len(views) for views in pixel_values_list)
        
        processed_views = []
        for view_idx in range(max_views):
            view_tensors = []
            for item_views in pixel_values_list:
                if view_idx < len(item_views):
                    view_tensors.append(item_views[view_idx])
                else:
                    first_valid_shape = next(tensor.shape for views in pixel_values_list for tensor in views)
                    view_tensors.append(
                        torch.zeros(
                            first_valid_shape,
                            dtype=next(iter(item_views)).dtype,
                            device=next(iter(item_views)).device
                        )
                    )
            processed_views.append(torch.stack(view_tensors))
        
        return {
            'pixel_values': processed_views,
            'input_ids': batch_text['input_ids'],
            'attention_mask': batch_text['attention_mask'],
            'labels': labels['input_ids']
        }

data_collator = DataCollatorForMultipleViews(tokenizer=tokenizer)
df = pd.read_csv('/project/lt200203-aimedi/pud/gen-x-report/mimic-run/fine-tuning-1/IU_MIMIC_Pad-GR.csv')
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = ChestXrayDataset(train_df, tokenizer, processor)
eval_dataset = ChestXrayDataset(eval_df, tokenizer, processor)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_model():
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.qformer.parameters():
        param.requires_grad = True
        
    for param in model.language_projection.parameters():
        param.requires_grad = True
        
    for param in model.language_model.model.embed_tokens.parameters():
        param.requires_grad = True
        
    for layer in model.language_model.model.layers:
        for param in layer.self_attn.parameters():
            param.requires_grad = True
        
    for param in model.language_model.lm_head.parameters():
        param.requires_grad = True
    #for k,v in dataset[0].items():
    #    print(f"{k} {v.shape} {v.dtype} {v.min()} {v.max()}")
    print("train:", len(train_dataset))
    print("eval:", len(eval_dataset))
    
    training_args = Seq2SeqTrainingArguments(
        output_dir='/scratch/lt200203-aimedi/save-v6/raddino-qformer-radllama-p3-t0.01-v6/checkpoints',
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_steps=128,#<--------
        num_train_epochs=10,#<--------
        save_steps=1,#<--------
        eval_steps=1,#<--------
        evaluation_strategy="epoch", 
        save_strategy="epoch",
        save_total_limit=10,
        logging_dir='/scratch/lt200203-aimedi/save-v6/raddino-qformer-radllama-p3-t0.01-v6/log',
        warmup_steps=7,#<--------
        warmup_ratio=1e-3,
        lr_scheduler_type='cosine',
        optim='adamw_torch',
        weight_decay=0.01,
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=True,
        disable_tqdm=False,
        dataloader_num_workers=8,
        load_best_model_at_end=True,
        dataloader_pin_memory=False,
        #predict_with_generate=True,
        #torch_compile=True,
        deepspeed= '/project/lt200203-aimedi/pud/gen-x-report/Encoder-Decoder/deepspeed_config.json'
    )
    trainer = accelerate.prepare(Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        #compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3,early_stopping_threshold=0.01)]
    ))
    trainer.train()
    trainer.save_model("/project/lt200203-aimedi/pud/gen-x-report/Encoder-Decoder/model-raddino-qformer-radllama-p3-t0.01-v6")
    trainer.save_model("/scratch/lt200203-aimedi/model-raddino-qformer-radllama-p3-t0.01-v6")
    
train_model()