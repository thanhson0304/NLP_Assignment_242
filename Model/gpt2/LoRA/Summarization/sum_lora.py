import random, numpy as np, torch, torch.nn as nn
from datasets import load_dataset
import evaluate
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.add_special_tokens({"additional_special_tokens": ["<sum>"]})
tok.pad_token = tok.eos_token
tok.padding_side = "left" 
print(f"Device: {device}")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.resize_token_embeddings(len(tok))

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["c_attn"],
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.to(device)
class GPT2Summarizer(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.lm_head = base.lm_head  # reuse LM head from base

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return {"loss": output.loss, "logits": output.logits}

    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)
train_raw = load_dataset("cnn_dailymail", "3.0.0", split="train[:100000]")
val_raw   = load_dataset("cnn_dailymail", "3.0.0", split="validation[:10000]")
rouge     = evaluate.load("rouge")

def preprocess(example):
    prefix = "<sum> " + example["article"]
    input_enc = tok(prefix, truncation=True, max_length=512, padding="max_length")
    label_enc = tok(example["highlights"], truncation=True, max_length=128, padding="max_length")
    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": label_enc["input_ids"]
    }

train_ds = train_raw.map(preprocess, remove_columns=train_raw.column_names)
val_ds = val_raw.map(preprocess, remove_columns=val_raw.column_names)

collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
args = TrainingArguments(
    output_dir="sum-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    fp16=(device == "cuda"),
    learning_rate=5e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False, 
    logging_steps=20,
    report_to=[]
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    tokenizer=tok
)

trainer.train()
model.save_pretrained("sum-lora")        
tok.save_pretrained("sum-lora") 
preds, refs = [], []
for ex in val_raw.select(range(10000)):
    inputs = tok(
        "<sum> " + ex["article"],
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True
    ).input_ids.to(device)

    output = model.generate(
        inputs,
        max_new_tokens=80,
        num_beams=4,
        length_penalty=1.2,
        early_stopping=True,
        pad_token_id=tok.eos_token_id
    )[0]

    summary = tok.decode(output[inputs.shape[1]:], skip_special_tokens=True).strip()
    preds.append(summary)
    refs.append(ex["highlights"])

result = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
print("Final ROUGE:", result)