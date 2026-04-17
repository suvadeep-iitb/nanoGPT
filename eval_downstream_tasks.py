import torch
import torch.nn.functional as F

from datasets import load_dataset
from tqdm import tqdm
import math, pickle
import numpy as np

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
batch_size = 16
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


@torch.no_grad()
def evaluate_ppl(dataset, batch_size=8, max_length=8192):
    total_loss = 0
    total_tokens = 0

    texts = [t for t in dataset["text"] if len(t.strip()) > 0]

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]

        # Tokenize with truncation/padding
        tokenizer.padding_side = "right"
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            device=device
        )

        input_ids = enc["input_ids"]
        targets = input_ids.clone()
        targets[:, :-1] = input_ids[:, 1:]

        if tokenizer.pad_token_id is not None:
            targets[targets == tokenizer.pad_token_id] = -1
        targets[:, -1] = -1

        with torch.no_grad():
            logits, loss = model(input_ids, targets=targets)

        if loss is not None:
            # multiply by number of non-pad tokens to get total loss
            batch_tokens = (targets != -1).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    ppl = math.exp(total_loss / total_tokens)
    return ppl


@torch.no_grad()
def evaluate_lambada(batch_size=128):
    dataset = load_dataset("lambada", split="test")

    contexts = []
    targets = []

    for ex in dataset:
        words = ex["text"].split()
        contexts.append(" ".join(words[:-1]) + " ")
        targets.append(words[-1])

    correct = 0
    total = len(contexts)

    for i in tqdm(range(0, total, batch_size)):
        batch_contexts = contexts[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]

        # concatenate context + target
        full_texts = [
            c + t for c, t in zip(batch_contexts, batch_targets)
        ]

        tokenizer.padding_side = "right"
        enc_full = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            device=device
        )

        enc_context = tokenizer(
            batch_contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            device=device
        )

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]

        # Create shifted targets
        targets_ids = input_ids.clone()
        targets_ids[:, :-1] = input_ids[:, 1:]
        targets_ids[:, -1] = -1

        if tokenizer.pad_token_id is not None:
            targets_ids[targets_ids == tokenizer.pad_token_id] = -1

        logits, _ = model(input_ids)

        log_probs = torch.log_softmax(logits, dim=-1)

        # Compute log-likelihood only over target span
        avg_log_probs = 0
        for b in range(len(batch_contexts)):

            context_len = enc_context["attention_mask"][b].sum().item()
            full_len = attention_mask[b].sum().item()

            # target tokens are from context_len-1 to full_len-1
            ll = 0
            for t in range(context_len-1, full_len-1):
                token_id = input_ids[b, t+1]
                ll += log_probs[b, t, token_id].item()
            avg_log_probs += ll / max(full_len-context_len, 1)

            # LAMBADA is single-answer → always correct if evaluated standalone
            # (you'd compare vs alternatives if you had distractors)

            # Here we check greedy consistency as fallback
            pred_tokens = []
            for t in range(context_len-1, full_len-1):
                pred_token = logits[b, context_len-1].argmax().item()
                pred_tokens.append(pred_token)
            pred_word = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

            if pred_word == batch_targets[b]:
                correct += 1

    avg_log_probs = avg_log_probs / total
    accuracy = correct / total

    print("LAMBADA likelihood:", avg_log_probs)
    print("LAMBADA accuracy:", accuracy)
    print()

    return avg_log_probs, accuracy


@torch.no_grad()
def evaluate_hellaswag(batch_size=128):
    dataset = load_dataset("hellaswag", split="validation")

    new_batch_size = batch_size // 4

    correct = 0
    total = len(dataset)

    for i in tqdm(range(0, total, new_batch_size)):
        batch = dataset[i:i+new_batch_size]

        contexts = batch["ctx"]
        all_endings = batch["endings"]  # list of list (B x 4)
        labels = [int(l) for l in batch["label"]] # Ensure they are ints

        # ---- Flatten (B x 4) → (B*4) ----
        flat_contexts = []
        flat_full = []

        for ctx, endings in zip(contexts, all_endings):
            ctx_clean = ctx.rstrip()
            for e in endings:
                flat_contexts.append(ctx.rstrip())
                flat_full.append(ctx.rstrip() + " " + e)

        # ---- Tokenize ----
        tokenizer.padding_side = "right"
        enc_full = tokenizer(
            flat_full,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            device=device
        )

        enc_ctx = tokenizer(
            flat_contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            device=device
        )

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]

        logits, _ = model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)

        # ---- Compute log-likelihoods ----
        scores = []

        for b in range(input_ids.size(0)):
            ctx_len = enc_ctx["attention_mask"][b].sum().item()
            full_len = attention_mask[b].sum().item()

            ll = 0.0
            for t in range(ctx_len - 1, full_len - 1):
                token_id = input_ids[b, t + 1]
                ll += log_probs[b, t, token_id].item()
            ll /= max(full_len - ctx_len, 1)
            scores.append(ll)

        # ---- Reshape back (B x 4) ----
        scores = torch.tensor(scores).view(len(contexts), 4)

        preds = scores.argmax(dim=1).tolist()
        labels = batch["label"]

        for p, l in zip(preds, labels):
            if p == l:
                correct += 1

    accuracy = correct / total
    print("HellaSwag accuracy:", accuracy)
    print()

    return accuracy


@torch.no_grad()
def evaluate_arc(batch_size=128):
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")

    new_batch_size = batch_size // 4

    correct = 0
    total = len(dataset)

    for i in tqdm(range(0, total, new_batch_size)):
        batch = dataset.select(range(i, min(i + new_batch_size, total)))

        flat_contexts = []
        flat_full = []
        choice_counts = []
        gold_indices = []

        # ---- Flatten ----
        for ex in batch:
            question = ex["question"].rstrip() + "\nAnswer:"
            choices = ex["choices"]["text"]
            labels = ex["choices"]["label"]

            # map answerKey to index
            gold_idx = labels.index(ex["answerKey"])
            gold_indices.append(gold_idx)

            choice_counts.append(len(choices))

            for c in choices:
                flat_contexts.append(question)
                flat_full.append(question + " " + c)

        # ---- Tokenize ----
        tokenizer.padding_side = "right"
        enc_full = tokenizer(
            flat_full,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            device=device
        )

        enc_ctx = tokenizer(
            flat_contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            device=device
        )

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]

        logits, _ = model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)

        # ---- Compute scores ----
        scores = []
        for b in range(input_ids.size(0)):
            ctx_len = enc_ctx["attention_mask"][b].sum().item()
            full_len = attention_mask[b].sum().item()

            ll = 0.0
            for t in range(ctx_len - 1, full_len - 1):
                token_id = input_ids[b, t + 1]
                ll += log_probs[b, t, token_id].item()

            # length normalization (important)
            length = max(full_len - ctx_len, 1)
            scores.append(ll / length)

        # ---- Unflatten ----
        idx = 0
        for b, num_choices in enumerate(choice_counts):
            choice_scores = scores[idx:idx+num_choices]
            pred = max(range(num_choices), key=lambda i: choice_scores[i])

            if pred == gold_indices[b]:
                correct += 1

            idx += num_choices

    accuracy = correct / total
    print("ARC-Challenge accuracy:", accuracy)
    print()

    return accuracy


@torch.no_grad()
def evaluate_mmlu(batch_size=128, subject="abstract_algebra"):
    dataset = load_dataset("cais/mmlu", subject, split="test")

    new_batch_size = batch_size // 4

    correct = 0
    total = len(dataset)

    for i in tqdm(range(0, total, new_batch_size)):
        batch = dataset.select(range(i, min(i + new_batch_size, total)))

        flat_contexts = []
        flat_full = []

        # ---- Flatten (B x 4) → (B*4) ----
        for ex in batch:
            question = ex["question"].rstrip() + "\nAnswer:"
            choices = ex["choices"]
            assert len(choices) == 4

            for c in choices:
                ctx = question
                full = ctx + " " + c
                flat_contexts.append(ctx)
                flat_full.append(full)

        # ---- Tokenize ----
        tokenizer.padding_side = "right"
        enc_full = tokenizer(
            flat_full,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            device=device
        )

        enc_ctx = tokenizer(
            flat_contexts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            device=device
        )

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]

        logits, _ = model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)

        # ---- Compute log-likelihoods ----
        scores = []
        for b in range(input_ids.size(0)):
            ctx_len = enc_ctx["attention_mask"][b].sum().item()
            full_len = attention_mask[b].sum().item()

            ll = 0.0
            for t in range(ctx_len - 1, full_len - 1):
                token_id = input_ids[b, t + 1]
                ll += log_probs[b, t, token_id].item()

            # length normalization (important)
            length = max(full_len - ctx_len, 1)
            scores.append(ll / length)

        # ---- Reshape (B*4 → B x 4) ----
        scores = torch.tensor(scores).view(len(batch), 4)

        preds = scores.argmax(dim=1).tolist()
        labels = batch["answer"]

        for p, l in zip(preds, labels):
            if p == l:
                correct += 1

    accuracy = correct / total
    print(subject, "accuracy:", accuracy)
    print()

    return accuracy


@torch.no_grad()
def evaluate_triviaqa(batch_size=128, max_new_tokens=32):
    # rc.nocontext is good for "closed-book" evaluation
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    # Pre-process: Get the question and ALL valid aliases
    questions = []
    gold_alias_lists = []
    for ex in dataset:
        questions.append(ex["question"].strip())
        # We want to check against 'value' AND all 'aliases'
        aliases = [ex["answer"]["value"]] + ex["answer"]["aliases"]
        gold_alias_lists.append([a.lower() for a in aliases])

    correct = 0
    total = len(questions)

    for i in tqdm(range(0, total, batch_size)):
        batch_q = questions[i : i + batch_size]
        batch_gold = gold_alias_lists[i : i + batch_size]

        # Completion-style prompt for base models
        prompts = [f"Question: {q}\nAnswer:" for q in batch_q]

        # Tokenize (using your wrapper or standard HF tokenizer)
        # Ensure padding_side is left for generation!
        tokenizer.padding_side = "left"
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            device=device
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Generate
        outputs = model.generate(
            idx=input_ids, # Use input_ids for HF models
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            top_k=1, # Greedy for evaluation
        )

        # Decode only the generated part
        for j in range(len(batch_q)):
            # Slice only new tokens
            gen_tokens = outputs[j][input_ids.shape[1]:]
            pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()

            # Check if any alias is in the prediction
            # Note: TriviaQA usually uses 'Exact Match', but for base models,
            # checking if the gold answer exists in the generated string is more realistic.
            if any(alias in pred_text for alias in batch_gold[j]):
                correct += 1

    accuracy = correct / total
    print(f"TriviaQA Accuracy: {accuracy}")
    return accuracy


def evaluate_pg19(batch_size=8):
    dataset = load_dataset("emozilla/pg19", split="test")

    ppl = evaluate_ppl(dataset, batch_size=batch_size, max_length=block_size)
    print("PG19 PPL:", ppl)
    print()

    return ppl


def evaluate_wikitext(batch_size=16):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    ppl = evaluate_ppl(dataset, batch_size=batch_size, max_length=block_size)
    print("WikiText103 PPL:", ppl)
    print()

    return ppl 


tokenizer = TiktokenWrapper()

ppl_wt = evaluate_wikitext()
ppl_pg19 = evaluate_pg19()
avg_lp_lambada, acc_lambada = evaluate_lambada()
acc_hellaswag = evaluate_hellaswag()
acc_arc = evaluate_arc()
acc_mmlu = evaluate_mmlu(subject="abstract_algebra")
acc_tqa = evaluate_triviaqa()

res_dic = {'ppl_wt': ppl_wt, 'ppl_pg19': ppl_pg19, 'avg_lp_lambada': avg_lp_lambada,
           'acc_lambada': acc_lambada, 'acc_hellaswag': acc_hellaswag, 
           'acc_arc': acc_arc, 'acc_mmlu': acc_mmlu, 'acc_tqa': acc_tqa}
save_file = f'downstream_task_results_{model_type}_blkSize{block_size}_{ckpt_idx}.pkl'

pickle.dump(res_dic, open(save_file, 'wb'))


