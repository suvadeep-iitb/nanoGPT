import torch
import torch.nn.functional as F

from datasets import load_dataset
from tqdm import tqdm
import math, pickle, collections
import numpy as np
import string

import torch
import tiktoken

import sys, os

seed = 42
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model_type = sys.argv[1]
ckpt_idx = int(sys.argv[2])

# Setup Model Parameters
device = 'cuda' 
dtype = 'bfloat16'
#model_type = 'softmax_rope'

block_size = 8192
#ckpt_idx = 8000
batch_size = 8
#ckpt_path = f'out/ckpt_softmaxLocalAttnAPE_L12_H12_HDim64_attnSpan100_blkSize{BLOCK_SIZE}_{ckpt_idx}.pt'

n_layer = 36
n_head = 20
head_dim = 64
n_embd = 1280
vocab_size = 50304
dropout = 0.0
bias = False

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

if model_type == 'softmax_ape':
    from model import GPTConfig, GPT

    model_args = dict(n_layer=n_layer, n_head=n_head, head_dim=head_dim, n_embd=n_embd, 
                  block_size=block_size, bias=bias, vocab_size=vocab_size, dropout=dropout)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif model_type == 'softmax_alibi':
    from model_alibi import GPTConfig, GPT

    ckpt_path = f'out/ckpt_softmaxALiBi_H36_HDim64_{ckpt_idx}.pt'

    model_args = dict(n_layer=n_layer, n_head=n_head, head_dim=head_dim, n_embd=n_embd, 
                  block_size=block_size, bias=bias, vocab_size=vocab_size, dropout=dropout)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif model_type == 'softmax_rope':
    from model_rope import GPTConfig, GPT

    ckpt_path = f'out/ckpt_softmaxRoPE_H36_HDim64_{ckpt_idx}.pt'
    rope_base = 10000 * (block_size // 2048)

    model_args = dict(n_layer=n_layer, n_head=n_head, head_dim=head_dim, n_embd=n_embd, 
                      rope_base=rope_base, block_size=block_size, bias=bias, 
                      vocab_size=vocab_size, dropout=dropout)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif model_type == 'softmaxLocalAttn_ape': 
    from model_localAttn import GPTConfig, GPT

    ckpt_path = f'out/ckpt_softmaxLocalAttn_H36_HDim64_{ckpt_idx}.pt'

    n_global_head = 1
    n_local_head = n_head - n_global_head
    local_attn_span = 100
    model_args = dict(n_layer=n_layer, n_embd=n_embd, n_head=n_head, head_dim=head_dim, 
                      n_local_head=n_local_head, local_attn_span=local_attn_span, 
                      block_size=block_size, bias=bias, vocab_size=vocab_size, 
                      dropout=dropout)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif model_type == 'softmaxLocalAttn_GH4_ape': 
    from model_localAttn import GPTConfig, GPT

    ckpt_path = f'out/models_softmaxLocalAttn_L36_H20_G4_HDim64/ckpt_softmaxLocalAttn_L36_G4_HDim64_{ckpt_idx}.pt'

    n_global_head = 4
    n_local_head = n_head - n_global_head
    local_attn_span = 100
    model_args = dict(n_layer=n_layer, n_embd=n_embd, n_head=n_head, head_dim=head_dim, 
                      n_local_head=n_local_head, local_attn_span=local_attn_span, 
                      block_size=block_size, bias=bias, vocab_size=vocab_size, 
                      dropout=dropout)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif model_type == 'softmaxLocalAttn_rope':
    raise ValueError(f"Unknown model_type: {model_type}")

else:
    raise ValueError(f"Unknown model_type: {model_type}")

checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    assert model_args[k] == checkpoint_model_args[k], (f'model arg: {model_args[k]}, checkpoint arg: {checkpoint_model_args[k]}')
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
if model_type == 'softmaxLocalAttn_ape':
    model = model.to(device, dtype=torch.bfloat16)
else:
    model = model.to(device)


class TiktokenWrapper:
    def __init__(self, encoding_name="gpt2", pad_token_id=None):
        self.enc = tiktoken.get_encoding(encoding_name)

        # GPT-2 default
        self.eot_token_id = self.enc.eot_token

        # define pad token
        self.pad_token_id = pad_token_id if pad_token_id is not None else self.eot_token_id

        # default behavior
        self.padding_side = "right"

        self.special_token_ids = {self.eot_token_id}

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        # Handle batch: if it's a list of lists, take the first one (or flatten)
        if len(token_ids) > 0 and isinstance(token_ids[0], list):
            # If you are decoding a batch, you might want to return a list of strings
            return [self.decode(t, skip_special_tokens) for t in token_ids]

        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in self.special_token_ids]

        return self.enc.decode(token_ids)

    def __call__(self,
                 texts,
                 return_tensors="pt",
                 padding=True,
                 truncation=True,
                 max_length=None,
                 device=None):

        # ---- Encode ----
        batch_ids = [self.encode(t) for t in texts]

        # ---- Truncate ----
        if truncation and max_length is not None:
            if self.padding_side == "right":
                # keep prefix
                batch_ids = [ids[:max_length] for ids in batch_ids]
            else:
                # keep suffix (important for generation!)
                batch_ids = [ids[-max_length:] for ids in batch_ids]

        # ---- Determine max length ----
        if padding:
            max_len = max(len(ids) for ids in batch_ids)
            if max_length is not None:
                max_len = min(max_len, max_length)
        else:
            max_len = None

        # ---- Pad ----
        input_ids = []
        attention_mask = []

        for ids in batch_ids:
            if padding:
                pad_len = max_len - len(ids)

                if self.padding_side == "right":
                    padded = ids + [self.pad_token_id] * pad_len
                    mask = [1] * len(ids) + [0] * pad_len
                else:  # left padding
                    padded = [self.pad_token_id] * pad_len + ids
                    mask = [0] * pad_len + [1] * len(ids)
            else:
                padded = ids
                mask = [1] * len(ids)

            input_ids.append(padded)
            attention_mask.append(mask)

        # ---- Convert to tensors ----
        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

            if device is not None:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


def normalize(text):
    """Standard normalization: lowercase, remove punctuation and extra whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = " ".join(text.split())
    return text

@torch.no_grad()
def evaluate_ruler(dataset_name="niah_single_1", context_length=1024, batch_size=8, max_new_tokens=32):
    try:
        dataset = load_dataset("json", data_files=f"../RULER/scripts/synthetic_datasets/{dataset_name}/validation.jsonl", split='train').select(range(500))
    except:
        print(f"Ensure you have the RULER jsonl file for {dataset_name} in your directory.")
        return

    total = len(dataset)
    model.eval()

    # Trackers: {depth_value: [list of booleans for correct/incorrect]}
    depth_results = collections.defaultdict(list)

    for i in tqdm(range(0, total, batch_size), desc=f"Evaluating {dataset_name}"):
        batch = dataset.select(range(i, min(i + batch_size, total)))

        batch_depths = []
        batch_prompts = []
        batch_gold_answers = []
        for ex in batch:
            prompt = ex["input"]
            gold_answers = ex["outputs"] if isinstance(ex["outputs"], list) else [ex["outputs"]]

            length = ex["length"]
            token_position_answer = ex.get("token_position_answer", 0)
            depth = ((token_position_answer / length) * 100) // 10

            batch_depths.append(depth)
            batch_prompts.append(prompt)
            batch_gold_answers.append(gold_answers)

        # 2. Tokenize
        tokenizer.padding_side = "left"
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=context_length - max_new_tokens,
            device=device
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # 3. Generate
        outputs = model.generate(
            idx=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            top_k=1,
        )

        # 4. Decode and Evaluate (This must be nested inside the batch loop)
        for j in range(len(batch)):
            gen_tokens = outputs[j][input_ids.shape[1]:]
            pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            pred_norm = normalize(pred_text)

            gold_answers = batch_gold_answers[j]
            depth = batch_depths[j]

            # --- CATEGORY SPECIFIC LOGIC ---

            # 1. SINGLE POINT RETRIEVAL (Single Needle, QA, and VAT)
            # Logic: If any one of the gold variations is in the prediction, it's a success.
            is_single_point = any(k in dataset_name for k in ["single", "qa", "vat"])
            if is_single_point:
                score = 1 if any(normalize(gold) in pred_norm for gold in gold_answers) else 0
                depth_results[depth].append(score)

            # 2. MULTI-RETRIEVAL & AGGREGATION
            elif any(k in dataset_name for k in ["multikey", "multivalue", "multiquery", "cwe", "fwe"]):
                gold_norm = [normalize(g) for g in gold_answers]
                matches = sum(1 for gold in gold_norm if gold in pred_norm)
                
                # Option A: Strict Binary (Only 1 if ALL found)
                # score = 1 if matches == len(gold_norm) else 0
                
                # Option B: Partial Credit / Recall (Better for 8k research analysis)
                score = matches / len(gold_norm) if len(gold_norm) > 0 else 0
                
                # Note: For CWE/FWE, 'depth' might be None. Ensure your depth_results dict handles None keys.
                depth_results[depth].append(score)

            # 3. FALLBACK
            else:
                # Default to partial match logic if dataset name is unexpected
                matches = sum(1 for gold in gold_answers if normalize(gold) in pred_norm)
                score = matches / len(gold_answers) if len(gold_answers) > 0 else 0
                depth_results[depth].append(score)

    # 5. Calculate Accuracy per Depth
    final_depth_stats = {}
    total_correct = 0
    total_samples = 0

    print("-" * 30)
    print(f"TASK: {dataset_name}")
    # Sorting ensures your output flows from 0% depth to 100% depth
    for d, results in sorted(depth_results.items()):
        acc = sum(results) / len(results)
        final_depth_stats[d] = acc
        print(f"Depth {d:>3} | Accuracy: {acc:.2f}")
        total_correct += sum(results)
        total_samples += len(results)

    # 6. Compute and Print Aggregate Accuracy
    aggregate_accuracy = total_correct / total_samples if total_samples > 0 else 0
    print("-" * 30)
    print(f"AGGREGATE   | Accuracy: {aggregate_accuracy:.2f} ({total_correct}/{total_samples})")
    print("-" * 30)
    print("\n"*4)

    return {
        "depth_stats": final_depth_stats,
        "aggregate": aggregate_accuracy
    }

tokenizer = TiktokenWrapper()

# Retrieval (NIAH Variations) 
# TASKS: cwe  fwe  niah_multikey_1  niah_multikey_2  niah_multikey_3  niah_multiquery  niah_multivalue  niah_single_1  niah_single_2  niah_single_3  qa_1  qa_2  vat
res_niah_single_1 = evaluate_ruler("niah_single_1", context_length=block_size, batch_size=batch_size, max_new_tokens=32) ## ***
res_niah_single_2 = evaluate_ruler("niah_single_2", context_length=block_size, batch_size=batch_size, max_new_tokens=32) ## ***
res_niah_single_3 = evaluate_ruler("niah_single_3", context_length=block_size, batch_size=batch_size, max_new_tokens=32) ## ***
# res_niah_multikey_1 = evaluate_ruler("niah_multikey_1", context_length=block_size, batch_size=batch_size, max_new_tokens=32) ## ***
# res_niah_multikey_2 = evaluate_ruler("niah_multikey_2", context_length=block_size, batch_size=batch_size, max_new_tokens=32) ## ***
# res_niah_multikey_3 = evaluate_ruler("niah_multikey_3", context_length=block_size, batch_size=batch_size, max_new_tokens=32) ## ***
# res_niah_multivalue = evaluate_ruler("niah_multivalue", context_length=block_size, batch_size=batch_size, max_new_tokens=32)
# res_niah_multiquery = evaluate_ruler("niah_multiquery", context_length=block_size, batch_size=batch_size, max_new_tokens=32)

# # Multi-hop Tracing (Variable Tracking)
# res_vt = evaluate_ruler("vt", context_length=block_size, batch_size=batch_size, max_new_tokens=32) ##**

# # Aggregation
# res_cwe = evaluate_ruler("cwe", context_length=block_size, batch_size=batch_size, max_new_tokens=32) ## ***
# res_fwe = evaluate_ruler("fwe", context_length=block_size, batch_size=batch_size, max_new_tokens=32)

# # Question Answering (QA)
# res_qa_1 = evaluate_ruler("qa_1", context_length=block_size, batch_size=batch_size, max_new_tokens=32) ## **
# res_qa_2 = evaluate_ruler("qa_2", context_length=block_size, batch_size=batch_size, max_new_tokens=32)


res_dic = {
    "res_niah_single_1": res_niah_single_1,
    "res_niah_single_2": res_niah_single_2,
    "res_niah_single_3": res_niah_single_3,
    # "res_niah_multikey_1": res_niah_multikey_1,
    # "res_niah_multikey_2": res_niah_multikey_2,
    # "res_niah_multikey_3": res_niah_multikey_3,
    # "res_niah_multivalue": res_niah_multivalue,
    # "res_niah_multiquery": res_niah_multiquery,
    # "res_vt": res_vt,
    # "res_cwe": res_cwe,
    # "res_fwe": res_fwe,
    # "res_qa_1": res_qa_1,
    # "res_qa_2": res_qa_2
}
save_file = f'rular_tasks_results_{model_type}_blkSize{block_size}_{ckpt_idx}.pkl'
pickle.dump(res_dic, open(save_file, 'wb'))



