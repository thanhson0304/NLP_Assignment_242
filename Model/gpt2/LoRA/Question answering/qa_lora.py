
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
class GPT2SpanQA(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.config = base.config
        H = base.config.hidden_size
        self.s = nn.Linear(H, 1)
        self.e = nn.Linear(H, 1)
    def forward(self, input_ids=None, attention_mask=None,
                start_positions=None, end_positions=None, labels=None,**kwargs):
        kwargs.pop("num_items_in_batch", None)
        hs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        ).last_hidden_state
        start = self.s(hs).squeeze(-1)
        end   = self.e(hs).squeeze(-1)
        loss  = None
        if start_positions is not None:
            ce  = nn.CrossEntropyLoss()
            loss = (ce(start,start_positions)+ce(end,end_positions))/2
        return {"loss": loss, "start_logits": start, "end_logits": end}

base_qa = GPT2Model.from_pretrained("gpt2")
qa_model = GPT2SpanQA(base_qa)
qa_lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    task_type=TaskType.TOKEN_CLS, target_modules=["c_attn"]
)
qa_model = get_peft_model(qa_model, qa_lora).to(device)


# Load the SQuAD dataset
def hr(t): print("\n" + "═"*15 + " " + t + " " + "═"*15)
hr("Training QA")
raw_qa = load_dataset("squad", split="train[:80000]")

def tok_qa(ex):
    q, c = ex["question"], ex["context"]
    a = ex["answers"]["text"][0]

    enc = tok(
        q + tok.eos_token + c,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )
    offsets = enc.pop("offset_mapping")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    idx = c.find(a)
    if idx == -1:
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": None,
                "end_positions": None}

    qlen = len(q) + len(tok.eos_token)
    start_char = qlen + idx
    end_char = start_char + len(a)

    start_pos = next(
        (i for i, (s, e) in enumerate(offsets) if s <= start_char < e),
        None
    )
    end_pos = next(
        (i for i, (s, e) in enumerate(offsets) if s < end_char <= e),
        None
    )

    if start_pos is None or end_pos is None:
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": None,
                "end_positions": None}

    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": start_pos,
            "end_positions": end_pos}

# Filter out examples without valid start positions
keep = ["input_ids", "attention_mask", "start_positions", "end_positions"]
train_qa = raw_qa.map(
    tok_qa,
    batched=False,
    remove_columns=[c for c in raw_qa.column_names if c not in keep]
)

train_qa = train_qa.filter(lambda ex: ex["start_positions"] is not None)

# Validation set preparation
val_raw = load_dataset("squad", split="validation[:5000]")
val_qa = val_raw.map(
    tok_qa,
    batched=False,
    remove_columns=val_raw.column_names
).filter(lambda ex: ex["start_positions"] is not None)

# Collate function to handle padding
def coll_qa(batch):
    input_ids = [torch.tensor(ex["input_ids"]) for ex in batch]
    masks = [torch.tensor(ex["attention_mask"]) for ex in batch]
    pad_id = tok.pad_token_id

    input_ids_padded = pad_sequence(
        input_ids, batch_first=True, padding_value=pad_id
    )
    masks_padded = pad_sequence(
        masks, batch_first=True, padding_value=0
    )
    starts = torch.tensor([ex["start_positions"] for ex in batch])
    ends = torch.tensor([ex["end_positions"] for ex in batch])

    return {
        "input_ids": input_ids_padded,
        "attention_mask": masks_padded,
        "start_positions": starts,
        "end_positions": ends,
    }

# Compute the metrics (EM and F1)
metric_squad = evaluate.load("squad")

def compute_metrics_qa(p):
    # Get predictions and labels
    preds = p.predictions
    labels = p.label_ids

    start_pred = preds[0]
    end_pred = preds[1]

    # Convert the start and end predictions to text
    decoded_preds = tok.decode(start_pred, skip_special_tokens=True)

    # Calculate EM and F1
    em = metric_squad.compute(predictions=decoded_preds, references=labels)["exact_match"]
    f1 = metric_squad.compute(predictions=decoded_preds, references=labels)["f1"]

    return {"eval_exact_match": em, "eval_f1": f1}

# TrainingArguments with matching eval & save strategy
args_qa = TrainingArguments(
    output_dir="qa-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    max_grad_norm=0.1,
    fp16=torch.cuda.is_available(),
    logging_strategy="steps",
    logging_steps=250,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to=[]
)

# Trainer setup with eval and metric computation
trainer_qa = Trainer(
    model=qa_model,
    args=args_qa,
    train_dataset=train_qa,
    eval_dataset=val_qa,          # Evaluation dataset
    data_collator=coll_qa,
    tokenizer=tok,
    compute_metrics=compute_metrics_qa  # Pass the compute_metrics function
)

# Train and save the model
trainer_qa.train()
qa_model.save_pretrained("qa-lora")
PeftModel.from_pretrained(GPT2Model.from_pretrained("gpt2"),
                         "qa-lora"
)


hr("Evaluation")

metric_squad = evaluate.load("squad")
val_qa = load_dataset("squad", split="validation[:5000]")

pred_list, ref_list = [], []
for i, ex in enumerate(tqdm.tqdm(val_qa, desc="QA")):
    enc = tok(ex["question"] + tok.eos_token + ex["context"],
              return_tensors="pt", truncation=True, max_length=512).to(device)
    outs = qa_model(**enc)
    s = int(torch.argmax(outs["start_logits"]))
    e = int(torch.argmax(outs["end_logits"]))
    ans = tok.decode(enc["input_ids"][0][s:e+1]).strip()

    pred_list.append({"id": str(i), "prediction_text": ans})
    ref_list.append({"id": str(i),
                     "answers": {"text": ex["answers"]["text"],
                                 "answer_start": []}})

print("QA EM / F1:",
      metric_squad.compute(predictions=pred_list, references=ref_list))