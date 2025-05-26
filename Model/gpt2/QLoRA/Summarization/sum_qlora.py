import random
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
import evaluate
from transformers import (
    GPT2TokenizerFast,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer, BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import tqdm

# ── 1) Reproducibility & Device ───────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

# ── 2) Tokenizer ───────────────────────────────────────────────────────────────
tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.add_special_tokens({"additional_special_tokens": ["<sum>"]})
tok.pad_token = tok.eos_token
tok.padding_side = "left"
# ── 3) Load & 4-bit Quantize Base Model ───────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=bnb_config,
    device_map="auto",
)
base.resize_token_embeddings(len(tok))
base = prepare_model_for_kbit_training(base)

# ── 4) Apply LoRA adapters ────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["c_attn"],
    task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(base, lora_cfg)
peft_model.print_trainable_parameters()
peft_model.to(device)

# ── 5) Wrap in your explicit head ─────────────────────────────────────────────
class GPT2Summarizer(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.lm_head = base_model.lm_head

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return {"loss": out.loss, "logits": out.logits}

    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)

model = GPT2Summarizer(peft_model)
train_raw = load_dataset("cnn_dailymail", "3.0.0", split="train[:50000]")
val_raw   = load_dataset("cnn_dailymail", "3.0.0", split="validation[:5000]")
rouge     = evaluate.load("rouge")

def preprocess(ex):
    inp = tok(
        "<sum> " + ex["article"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    lbl = tok(
        ex["highlights"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    return {
        "input_ids":      inp["input_ids"],
        "attention_mask": inp["attention_mask"],
        "labels":         lbl["input_ids"],
    }

train_ds = train_raw.map(preprocess, batched=False, remove_columns=train_raw.column_names)
val_ds   = val_raw.map(preprocess,   batched=False, remove_columns=val_raw.column_names)

# ── 7) Data collator ──────────────────────────────────────────────────────────
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
# ── 8) Training arguments ─────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir="sum-qlora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    fp16=(device=="cuda"),
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    logging_steps=20,
    save_strategy="no",
    report_to=[],
)

# ── 9) Trainer setup ─────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    tokenizer=tok,
)

# ── 10) Train the QLoRA adapter ───────────────────────────────────────────────
trainer.train()
model.base.save_pretrained("sum-qlora")  
tok.save_pretrained("sum-qlora")
preds, refs = [], []
for ex in val_raw.select(range(5000)):
    inputs = tok(
        "<sum> " + ex["article"],
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).input_ids.to(device)

    output = peft_model.generate(
        inputs,
        max_new_tokens=80,
        num_beams=4,
        length_penalty=1.2,
        early_stopping=True,
        pad_token_id=tok.eos_token_id,
    )[0]

    summary = tok.decode(output[inputs.shape[1]:], skip_special_tokens=True).strip()
    preds.append(summary)
    refs.append(ex["highlights"])

result = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
print("Final ROUGE:", result)