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
from sklearn.model_selection import train_test_split

accelerate = Accelerator()

encoder_path = '/project/lt200203-aimedi/pud/gen-x-report/model/rad-dino-12c'
deocder_path = '/project/lt200203-aimedi/pud/gen-x-report/model/RadLLaMA-7b'

model = Blip2ForConditionalGeneration.from_pretrained("/project/lt200203-aimedi/pud/gen-x-report/model/blip2-opt-2.7b",ignore_mismatched_sizes=True)

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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load images
        view1 = self._safe_open_image(row.get('view1'))
        view2 = self._safe_open_image(row.get('view2'))
        view3 = self._safe_open_image(row.get('view3'))

        # Process images
        view1_img_processed = (
            self.processor(images=view1, return_tensors="pt", do_rescale=False)['pixel_values']
            if view1 else torch.zeros((3, 518, 518))
        )
        view2_img_processed = (
            self.processor(images=view2, return_tensors="pt", do_rescale=False)['pixel_values']
            if view2 else torch.zeros((3, 518, 518))
        )
        view3_img_processed = (
            self.processor(images=view3, return_tensors="pt", do_rescale=False)['pixel_values']
            if view3 else torch.zeros((3, 518, 518))
        )

        combined_img = torch.cat([
            view1_img_processed.squeeze(0) if view1 else torch.zeros(3, 518, 518),
            view2_img_processed.squeeze(0) if view2 else torch.zeros(3, 518, 518),
            view3_img_processed.squeeze(0) if view3 else torch.zeros(3, 518, 518),
        ], dim=0)

        # Encode text
        text = f"{row['findings']} [SEP] {row['impression']}"
        encodings = self.tokenizer(text)
        encodings.input_ids.append(self.tokenizer.eos_token_id)
        encodings.attention_mask.append(1)

        return {
            'combined_img': combined_img,
            'input_ids': torch.tensor(encodings['input_ids']).squeeze(0),
            'attention_mask': torch.tensor(encodings['attention_mask']).squeeze(0),
            'labels': torch.tensor(encodings['input_ids']).squeeze(0)
        }

#import warnings
#warnings.filterwarnings(
#    "ignore",
#    message="You are using `torch.load` with `weights_only=False`",
#    category=FutureWarning,
#)
#
#class PreprocessedDataset(Dataset):
#    
#    def __init__(self, save_dir, num_to_load=None):
#        
#        print(f"Initializing dataset from {save_dir}...")
#        self.file_paths = sorted(
#            [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.pt')]
#        )
#        if num_to_load is not None:
#            self.file_paths = self.file_paths[:num_to_load]
#        self.data = []
#
#    def __len__(self):
#        
#        return len(self.file_paths)
#
#    def __getitem__(self, idx):
#        
#        file_path = self.file_paths[idx]
#        try:
#            item = torch.load(file_path, map_location='cpu')
#            return item
#        except Exception as e:
#            print(f"Skipping corrupted file: {file_path}, Error: {e}")
#            #raise IndexError(f"Item at index {idx} could not be loaded.")


#def data_collator(batch):
#    input_ids = torch.stack([item.get('input_ids', torch.zeros(518, dtype=torch.long)) for item in batch]).to(device)
#    attention_mask = torch.stack([item.get('attention_mask', torch.zeros(518, dtype=torch.long)) for item in batch]).to(device)
#    labels = torch.stack(item[labels]) for item in batch]).to(device)
#    pixel_values = torch.stack([item["pixel_values"] for item in batch]).to(device)

def data_collator(batch):
    inputs = tokenizer.pad(
        [{"input_ids": item["input_ids"], 
        "attention_mask": item["attention_mask"]} for item in batch],
        padding=True,
        return_tensors="pt"
    )
    pixel_values = torch.stack([item["combined_img"] for item in batch])
    
    return {
        'pixel_values': pixel_values,
        'input_ids': inputs["input_ids"],
        'attention_mask': inputs["attention_mask"],
        'labels': inputs["input_ids"],
    }

df = pd.read_csv('/project/lt200203-aimedi/pud/gen-x-report/mimic-run/fine-tuning-1/cleaned-2000-case.csv')
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = ChestXrayDataset(train_df, tokenizer, processor)
eval_dataset = ChestXrayDataset(eval_df, tokenizer, processor)

#train_dataset = PreprocessedDataset(
#    save_dir="/scratch/lt200203-aimedi/mimic-cxr-tencor/train",
#    num_to_load=None
#    )
#eval_dataset = PreprocessedDataset(
#    save_dir="/scratch/lt200203-aimedi/mimic-cxr-tencor/eval",
#    num_to_load=None
#    )

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_model():
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.vision_model.embeddings.parameters():
        param.requires_grad = True
        
    #for param in model.vision_model.linear_proj.parameters():
    #    param.requires_grad = True
    #    
    #for param in model.vision_model.layernorm.parameters():
    #    param.requires_grad = True
        
    for param in model.qformer.parameters():
        param.requires_grad = True
        
    for param in model.language_projection.parameters():
        param.requires_grad = True
        
    for param in model.language_model.model.embed_tokens.parameters():
        param.requires_grad = True
        
    for layer in model.language_model.model.layers:
        for param in layer.self_attn.parameters():
            param.requires_grad = True
    #for k,v in dataset[0].items():
    #    print(f"{k} {v.shape} {v.dtype} {v.min()} {v.max()}")
    print("train:", len(train_dataset))
    print("eval:", len(eval_dataset))
    
    training_args = Seq2SeqTrainingArguments(
        output_dir='/scratch/lt200203-aimedi/gen-x-report/raddino-qformer-radllama-p3-t0.01-test-v4/checkpoints',
        per_device_eval_batch_size=16,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_steps=6,#<--------
        num_train_epochs=5,#<--------
        save_steps=1,#<--------
        eval_steps=1,#<--------
        evaluation_strategy="epoch", 
        save_strategy="epoch",
        save_total_limit=10,
        logging_dir='/scratch/lt200203-aimedi/gen-x-report/raddino-qformer-radllama-p3-t0.01-test-v4/log',
        warmup_steps=1,#<--------
        warmup_ratio=1e-3,
        lr_scheduler_type='cosine',
        optim='adamw_torch',
        weight_decay=0.01,
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=True,
        disable_tqdm=True,
        dataloader_num_workers=8,
        load_best_model_at_end=True,
        dataloader_pin_memory=False,
        predict_with_generate=True,
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
    trainer.save_model("/scratch/lt200203-aimedi/gen-x-report/model-raddino-qformer-radllama-p3-t0.01-test-v4")
    
train_model()