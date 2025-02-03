from transformers import (
    AutoImageProcessor, 
    AutoModel,
    Seq2SeqTrainer,
    EarlyStoppingCallback, 
    Seq2SeqTrainingArguments,
    AutoTokenizer,
                            )
import torch
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

import sys
sys.path.append("/project/lt200203-aimedi/pud/gen-x-report/modeling")

from gen_x_report.configuration_gen_x_report import GenXReportConfig
from gen_x_report.modeling_gen_x_report import GenXReportModel

GenXReportConfig.register_for_auto_class()
GenXReportModel.register_for_auto_class("AutoModel")

from gen_x_report.configuration_gen_x_report import GXRVisionConfig, GXRQFormerConfig, GXRTextConfig, GenXReportConfig

# Create instances of the sub-configurations
vision_config = GXRVisionConfig()
qformer_config = GXRQFormerConfig()
text_config = GXRTextConfig()

# Create the main configuration by passing the sub-configurations directly
genxreport_config = GenXReportConfig(
    vision_config=vision_config,
    qformer_config=qformer_config,
    text_config=text_config,
    num_query_tokens=32
)

model = AutoModel.from_pretrained(
    '/project/lt200203-aimedi/pud/gen-x-report/modeling/model', 
    config = genxreport_config, 
    trust_remote_code = True
    )
processor = AutoImageProcessor.from_pretrained('/project/lt200203-aimedi/pud/gen-x-report/model/rad-dino-12c')
tokenizer = AutoTokenizer.from_pretrained('/project/lt200203-aimedi/pud/gen-x-report/model/RadLLaMA-7b',trust_remote_code=True)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["[SEP]"],
})
tokenizer.pad_token = '[PAD]'
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# for reduce_bucket_size in deepspeedconfig
model.config.hidden_size = model.config.text_config.hidden_size

model.config.text_config.vocab_size = 32065
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

        processed_views = [None, None]

        # Process Frontal image
        frontal_path = row.get('Frontal_path')
        if frontal_path:
            frontal_view = self._safe_open_image(frontal_path)
            if frontal_view is not None:
                processed_views[0] = self._process_image(frontal_view)

        # Process Lateral image
        lateral_path = row.get('Lateral_path')
        if lateral_path:
            lateral_view = self._safe_open_image(lateral_path)
            if lateral_view is not None:
                processed_views[1] = self._process_image(lateral_view)

        # Skip if both images are missing
        if all(view is None for view in processed_views):
            print(f"Skipping index {idx} due to no valid images.")
            return self.__getitem__((idx + 1) % len(self.df))

        # Replace missing images with blank tensors if needed
        for i in range(2):
            if processed_views[i] is None:
                processed_views[i] = torch.zeros_like(processed_views[0] if processed_views[0] is not None else torch.zeros((3, 518, 518)))

        # Encode text with tokenizer
        text = f"{row['findings']} [SEP] {row['impression']}"
        encodings = self.tokenizer(text)
        labels = encodings['input_ids'] + [self.tokenizer.eos_token_id]
        input_ids = encodings['input_ids'] + [self.tokenizer.eos_token_id]

        return {
            'pixel_values': processed_views,
            'input_ids': input_ids,
            #'attention_mask': attention_mask,
            'labels': labels
        }

@dataclass
class DataCollatorForMultipleViews:
    tokenizer: Any
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values_list = [item.pop('pixel_values') for item in features]

        input_ids = self.tokenizer.pad(
            [{'input_ids': f['input_ids']} for f in features],
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"]

        labels = self.tokenizer.pad(
            [{'input_ids': f['labels']} for f in features],
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"]

        frontal_views = torch.stack([views[0] for views in pixel_values_list])
        lateral_views = torch.stack([views[1] for views in pixel_values_list])

        return {
            'pixel_values': [frontal_views, lateral_views],
            'input_ids': input_ids,
            'labels': labels
        }

data_collator = DataCollatorForMultipleViews(tokenizer=tokenizer)
df = pd.read_csv('/project/lt200203-aimedi/pud/gen-x-report/preparedata_iuxray.csv')
#df = df.sample(n=2000, random_state=42)
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = ChestXrayDataset(train_df, tokenizer, processor)
eval_dataset = ChestXrayDataset(eval_df, tokenizer, processor)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_model():
    for param in model.vision_model.parameters():
        param.requires_grad = False

    for layer in model.language_model.model.layers:
        for param in layer.mlp.parameters():
            param.requires_grad = False

    for layer in model.language_model.model.layers:
        for param in layer.input_layernorm.parameters():
            param.requires_grad = False

    for layer in model.language_model.model.layers:
        for param in layer.post_attention_layernorm.parameters():
            param.requires_grad = False

    for param in model.language_model.model.norm.parameters():
        param.requires_grad = False

    for param in model.language_model.model.rotary_emb.parameters():
        param.requires_grad = False

    #for k,v in dataset[0].items():
    #    print(f"{k} {v.shape} {v.dtype} {v.min()} {v.max()}")
    print("train:", len(train_dataset))
    print("eval:", len(eval_dataset))
    
    training_args = Seq2SeqTrainingArguments(
        output_dir='/project/lt200203-aimedi/pud/gen-x-report/modeling/train/v2/checkpoints',
        per_device_eval_batch_size=20,
        per_device_train_batch_size=20,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_steps=24,#<--------
        num_train_epochs=5,#<--------
        save_steps=1,#<--------
        eval_steps=1,#<--------
        evaluation_strategy="epoch", 
        save_strategy="epoch",
        save_total_limit=10,
        logging_dir='/project/lt200203-aimedi/pud/gen-x-report/modeling/train/v2/log',
        warmup_steps=5,#<--------
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
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        #compute_metrics=compute_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=2,early_stopping_threshold=0.01)]
    )
    trainer.train()
    trainer.save_model("/project/lt200203-aimedi/pud/gen-x-report/modeling/train/v2/model")

train_model()