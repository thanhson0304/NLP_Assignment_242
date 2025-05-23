import torch, torch.nn as nn, random, numpy as np, evaluate
from datasets import load_dataset
from peft import PeftModel
from torch.nn.utils.rnn import pad_sequence
import tqdm
from transformers import GPT2Model
from transformers import (
    GPT2TokenizerFast, GPT2Model, GPT2LMHeadModel,
    TrainingArguments, Trainer,BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import classification_report

device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

tok = GPT2TokenizerFast.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
def make_qlora_model(base_name, head_cls, lora_r=8, task_type="SEQ_CLS"):
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    if head_cls is GPT2LMHeadModel:
        base = GPT2LMHeadModel.from_pretrained(
            base_name, quantization_config=qconf, device_map="auto"
        )
    else:
        base = GPT2Model.from_pretrained(
            base_name, quantization_config=qconf, device_map="auto"
        )
    model = head_cls(base) if head_cls is not GPT2LMHeadModel else base

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type=getattr(TaskType, task_type),
        target_modules=["c_attn"],
    )
    model = get_peft_model(model, lora_cfg).to(device)
    return model

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

sent_model = make_qlora_model("gpt2", GPT2Sentiment, lora_r=8, task_type="SEQ_CLS")
def hr(t): print("\n" + "═"*15 + " " + t + " " + "═"*15)

hr("Training Sentiment")
raw = load_dataset("amazon_polarity")

def tok_sent(ex):
    return tok(ex["title"] + " " + ex["content"],
               truncation=True, max_length=256)

train_s = raw["train"].shuffle(seed=SEED).select(range(200000)).map(tok_sent)
val_s   = raw["test"] .select(range(20000)).map(tok_sent)
train_s = train_s.rename_column("label", "labels")
val_s   = val_s.rename_column("label", "labels")
args_s = TrainingArguments(
    output_dir="sent-qlora",
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
sent_model.save_pretrained("sent-qlora")
base = GPT2Model.from_pretrained("gpt2")

sent_adapter = PeftModel.from_pretrained(base, "sent-qlora")
hr("Evaluation")

sent_val = load_dataset("amazon_polarity", split="test[:20000]")
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
print("Classification Report:")
print(classification_report(refs, preds, target_names=["negative","positive"]))
