import torch
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import numpy as np
from contextlib import nullcontext

seed = 42
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Ensures deterministic algorithms where possible
#torch.use_deterministic_algorithms(True)

# cuDNN settings (important)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


#####################################################################################################################
#####################################################################################################################
# Setup Scorers
r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
b_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

#####################################################################################################################
#####################################################################################################################
# Define NIAH Parameters
DEPTH_PERCENTS = [0, 25, 50, 75, 100] # Location of needle
MAX_NEW_TOKENS = 5
BLOCK_SIZE = 8192
N_TRIALS = 8
NEEDLE_REPEAT = 1
model_type = 'softmax_rope'

#####################################################################################################################
#####################################################################################################################
# Setup Tokenizer
import sentencepiece as spm
tokenizer = spm.SentencePieceProcessor(model_file="data/pg19_sentPieceTokenizer_vocabSize10K/pg19_D2800.model")

#####################################################################################################################
#####################################################################################################################
# Setup Model Parameters
device = 'cuda' 
dtype = 'bfloat16' 

ckpt_idx = 16000
ckpt_path = f'out/ckpt_softmaxRoPE_H12_L12_HDim64_blkSize{BLOCK_SIZE}_{ckpt_idx}.pt'
data_path = 'data/pg19_sentPieceTokenizer_vocabSize10K/sent_tokenized_pg19_validation.pt'

block_size = BLOCK_SIZE
n_layer = 12
n_head = 12
head_dim = 64
n_embd = 768
vocab_size = 10000
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

elif model_type == 'softmax_rope':
    from model_rope import GPTConfig, GPT

    rope_base = 10000 * (block_size // 2048)

    model_args = dict(n_layer=n_layer, n_head=n_head, head_dim=head_dim, n_embd=n_embd, 
                      rope_base=rope_base, block_size=block_size, bias=bias, 
                      vocab_size=vocab_size, dropout=dropout)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif model_type == 'softmaxLocalAttn_ape': 
    from model_localAttn import GPTConfig, GPT

    n_global_head = 1
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
model.to(device)

#####################################################################################################################
#####################################################################################################################
# Setup Dataset
filler_data = torch.load(data_path)


def get_context(needle_tokens, question_tokens, filler_data, n_trials, depth, needle_repeat, context_length):
    repeated_needle = torch.concat([needle_tokens]*needle_repeat, dim=0)
    filler_length = context_length - repeated_needle.shape[0] - question_tokens.shape[0]

    def get_filler_context(filler_data, filler_length):
        filler_tokens = filler_data['input_ids']
        doc_idx = filler_data['doc_idx']
        sent_idx = filler_data['sent_idx']

        filler_context = []
        filler_sent = []
        cur_filler_length = 0
        while cur_filler_length < filler_length:
            doc_id = torch.randint(0, len(doc_idx)-1, (1,)).item()
            doc_start = doc_idx[doc_id]
            doc_end = doc_idx[doc_id+1]
            filler_context.append(filler_tokens[doc_start: doc_end])

            start_sent_idx = torch.searchsorted(sent_idx, doc_start)
            end_sent_idx = torch.searchsorted(sent_idx, doc_end)
            filler_sent.append(sent_idx[start_sent_idx+1: end_sent_idx+1] - sent_idx[start_sent_idx] + cur_filler_length)

            cur_filler_length += (doc_end - doc_start)
        filler_context = torch.concat(filler_context, dim=0)
        filler_sent = torch.concat(filler_sent, dim=0)

        if cur_filler_length > filler_length:
            filler_context = filler_context[-filler_length:]
            filler_sent = filler_sent - (cur_filler_length - filler_length)
            filler_sent = filler_sent[filler_sent>0]
        filler_sent = torch.concat([torch.tensor([0]), filler_sent], dim=0)

        return filler_context.to(device), filler_sent.to(device)

    context_with_needle_list = []
    context_without_needle_list = []
    for t in range(n_trials):
        context_without_needle, sent_idx = get_filler_context(filler_data, filler_length)
        if depth == 0:
            context_with_needle = torch.concat([repeated_needle, context_without_needle], dim=0)
        elif depth == 100:
            context_with_needle = torch.concat([context_without_needle, repeated_needle], dim=0)
        else:
            insertion_idx = int(context_without_needle.shape[0] * (depth / 100))
            idx = torch.searchsorted(sent_idx, insertion_idx, right=True)
            insertion_idx = sent_idx[idx] if (sent_idx[idx] - insertion_idx) < (insertion_idx - sent_idx[idx-1]) else sent_idx[idx-1]
            context_with_needle = torch.concat(
                [context_without_needle[:insertion_idx], repeated_needle, context_without_needle[insertion_idx:]],
                dim = 0
            )
        context_without_needle = torch.concat([context_without_needle, question_tokens], dim=0)
        context_without_needle_list.append(context_without_needle)

        context_with_needle = torch.concat([context_with_needle, question_tokens], dim=0)
        context_with_needle_list.append(context_with_needle)
    context_without_needle = torch.stack(context_without_needle_list, dim=0)
    context_with_needle = torch.stack(context_with_needle_list, dim=0)
                
    return context_without_needle, context_with_needle


def calculate_answer_perplexity(context, answer):
    if answer.dim() == 1:
        answer = answer.unsqueeze(0)
    if answer.size(0) == 1 and context.size(0) > 1:
        answer = answer.repeat(context.size(0), 1)

    # 2. Combine them to get the full sequence
    full_ids = torch.cat([context, answer], dim=-1)
    full_ids = full_ids.to(device)

    # Target indices: We only want to calculate loss on the 'answer' portion
    # In Causal LM, the logit at index i predicts the token at index i+1
    target_len = answer.shape[-1]

    with torch.no_grad(), ctx:
        logits, _ = model(full_ids)

    # 3. Align Logits and Targets
    # We need the logits that correspond to the positions predicting the answer
    # If full_ids has length N, and ans_ids has length M:
    # The first token of the answer is at index N-M.
    # It is predicted by the logit at index N-M-1.
    shift_logits = logits[..., -(target_len + 1):-1, :].contiguous()
    shift_labels = full_ids[..., -target_len:].contiguous()

    # 4. Calculate Loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    batch_losses = token_losses.view(shift_logits.size(0), -1).mean(dim=1)
    batch_ppls = torch.exp(batch_losses).cpu()

    return batch_ppls


def evaluate_niah(needle_data, filler_data, n_trials, depth, NEEDLE_REPEAT, BLOCK_SIZE, MAX_NEW_TOKENS):
    # Place needle at 'depth' and append the completion prompt
    needle = needle_data["n"]
    question = needle_data["q"]
    answer = needle_data["a"]
            
    needle_tokens = torch.tensor(tokenizer.encode(needle)).to(device)
    question_tokens = torch.tensor(tokenizer.encode(question)).to(device)
    answer_tokens = torch.tensor(tokenizer.encode(answer)).to(device)

    max_context_length = BLOCK_SIZE - max(MAX_NEW_TOKENS, answer_tokens.shape[0])
    context_without_needle, context_with_needle = get_context(needle_tokens, question_tokens, filler_data, n_trials, 
                                                              depth, NEEDLE_REPEAT, max_context_length)

    ppls_with_needle = calculate_answer_perplexity(context_with_needle, answer_tokens)
    ppls_without_needle = calculate_answer_perplexity(context_without_needle, answer_tokens)

    retrieval_gains = ppls_without_needle / ppls_with_needle

    # Generate Completion
    context_with_needle = context_with_needle.to(device)
    context_length = context_with_needle.shape[-1]
    with torch.no_grad(), ctx:
        output_ids = model.generate(context_with_needle, max_new_tokens=MAX_NEW_TOKENS, temperature=1.0, top_k=1)
    generated_texts = [
        tokenizer.decode(ids.tolist())
        for ids in output_ids[:, context_length:]
    ]

    # Compute Metrica
    rouge_scores = []
    for gen in generated_texts:
        score = r_scorer.score(answer, gen)['rougeL'].fmeasure
        rouge_scores.append(score)
    rouge_scores = torch.tensor(rouge_scores)

    P, R, F1 = b_scorer.score(generated_texts, [answer]*len(generated_texts))
    bert_f1 = F1
    
    results = {
        "generated": generated_texts,
        "rouge_scores": rouge_scores,
        "bert_scores": bert_f1,
        "ppls_with_needle": ppls_with_needle,
        "ppls_without_needle": ppls_without_needle,
        "retrieval_gains": retrieval_gains
    }
            
    return results


# Pool of Victorian-style needles
NEEDLE_POOL = [
    {"n": " The Admiral’s favorite walking stick was carved from the bone of a narwhal. ", "q": " Based on the description of the study, the Admiral’s favorite walking stick was carved from the bone of a ", "a": "narwhal"},
    {"n": " The countess kept her most private letters in a folder of violet leather. ", "q": " Based on the description of the passage, the countess kept her most private letters in a folder of ", "a": "violet leather"},
    {"n": " Beneath the main hall lay a sanctuary of forgotten knowledge that few were permitted to enter. The old library was said to contain exactly seventy-four shelves of forbidden poetry. ", "q": " According to the accounts of the sanctuary, the old library was said to contain exactly ", "a": "seventy-four shelves"},
    {"n": " The clerk signed the final ledger using a quill dipped in indigo ink. ", "q": " At the conclusion of the documentation, the clerk signed the final ledger using a quill dipped in ", "a": "indigo ink"},
]


def run_robust_eval():
    for depth in DEPTH_PERCENTS:
        for n, needle_data in enumerate(NEEDLE_POOL):
            results = evaluate_niah(needle_data, filler_data, N_TRIALS, depth, NEEDLE_REPEAT, BLOCK_SIZE, MAX_NEW_TOKENS)
            results['depth'] = depth
            results['needle_data'] = needle_data
            results['block_size'] = block_size
            results['model_type'] = model_type
            results['model_args'] = model_args

            if model_type == 'softmax_ape':
                save_file = f'res_niah_depth{depth}_needle{n}_blockSize{BLOCK_SIZE}_{model_type}_L{n_layer}_H{n_head}_{ckpt_idx}.pt'
            elif model_type == 'softmax_rope':
                save_file = f'res_niah_depth{depth}_needle{n}_blockSize{BLOCK_SIZE}_{model_type}_L{n_layer}_H{n_head}_rbase{rope_base}_{ckpt_idx}.pt'
            elif model_type == 'softmaxLocalAttn_ape':
                save_file = f'res_niah_depth{depth}_needle{n}_blockSize{BLOCK_SIZE}_{model_type}_L{n_layer}_H{n_head}_local{n_local_head}_global{n_global_head}_{ckpt_idx}.pt'
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            torch.save(results, save_file)


run_robust_eval()

