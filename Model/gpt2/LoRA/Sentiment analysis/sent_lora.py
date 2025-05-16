import torch, torch.nn as nn, random, numpy as np, evaluate
from datasets import load_dataset
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
import tqdm
from transformers import GPT2Model
from transformers import (
    GPT2TokenizerFast, GPT2Model, GPT2LMHeadModel,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
class GPT2Sentiment(nn.Module):
    def __init__(self, base, n_labels=2):
        super().__init__()
        self.base = base
        self.config = base.config
        self.dropout = nn.Dropout(0.1)
        self.cls = nn.Linear(base.config.hidden_size, n_labels)
    def forward(self, input_ids=None, attention_mask=None,
                labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        hs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        ).last_hidden_state

        logits = self.cls(self.dropout(hs[:, -1]))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}

base_cls = GPT2Model.from_pretrained("gpt2")
sent_model = GPT2Sentiment(base_cls)
sent_lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    task_type=TaskType.SEQ_CLS, target_modules=["c_attn"]
)
sent_model = get_peft_model(sent_model, sent_lora).to(device)

def hr(t): print("\n" + "═"*15 + " " + t + " " + "═"*15)

# ── Sentiment (Amazon Polarity) ──────────────────────────────
hr("Training Sentiment")
raw = load_dataset("amazon_polarity")

def tok_sent(ex):
    return tok(ex["title"] + " " + ex["content"],
               truncation=True, max_length=256)

train_s = raw["train"].shuffle(seed=SEED).select(range(100_000)).map(tok_sent)
val_s   = raw["test"] .select(range(20_000)).map(tok_sent)
train_s = train_s.rename_column("label", "labels")
val_s   = val_s.rename_column("label", "labels")

args_s = TrainingArguments(
    output_dir="sent-lora",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=0.1,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to=[]
)

Trainer(sent_model, args_s,
        train_dataset=train_s,
        eval_dataset=val_s,
        tokenizer=tok).train()
sent_model.save_pretrained("sent-lora")
base = GPT2Model.from_pretrained("gpt2")

sent_adapter = PeftModel.from_pretrained(base, "sent-lora")

hr("Evaluation")

sent_val = load_dataset("amazon_polarity", split="test[:10000]")
preds, refs = [], []
for ex in tqdm.tqdm(sent_val, desc="Sentiment"):
    enc = tok(ex["title"] + " " + ex["content"],
              return_tensors="pt", truncation=True, max_length=256).to(device)
    logits = sent_model(**enc)["logits"]
    preds.append(int(logits.argmax(-1)))
    refs.append(ex["label"])

acc = evaluate.load("accuracy").compute(predictions=preds, references=refs)["accuracy"]
f1  = evaluate.load("f1").compute(predictions=preds, references=refs)["f1"]
print(f"Sentiment –  Acc: {acc:.4f}   F1: {f1:.4f}")