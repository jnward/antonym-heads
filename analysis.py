# %%

# Standard imports
import os
import torch
from tqdm import tqdm
import plotly.express as px

# Imports for displaying vis in Colab / notebook
import webbrowser
import http.server
import socketserver
import threading
PORT = 8000


torch.set_grad_enabled(False)

# %%

from transformer_lens import HookedTransformer

device = 'mps'

model = HookedTransformer.from_pretrained(
    # "gpt2-xl",
    "EleutherAI/pythia-1.4b",
    # "meta-llama/Llama-3.2-1B",
    fold_ln=True,
    device=device
)

# %%
model = model.to(torch.bfloat16)

# %%
tokenizer = model.tokenizer

# %%
import plotly.express as px

# adds a preceeding space
def tokenize(str):
    return tokenizer.encode(f" {str}", add_special_tokens=False)


# %%
pairs = [
    ('true', 'false'),
    ('hot', 'cold'),
    ('tall', 'short'),
    ('big', 'small'),
    ('open', 'closed'),
    ('happy', 'sad'),
    ('light', 'dark'),
    ('fast', 'slow'),
    ('strong', 'weak'),
    ('hard', 'soft'),
    ('early', 'late'),
    ('full', 'empty'),
    ('high', 'low'),
    ('inside', 'outside'),
    ('clean', 'dirty'),
    ('new', 'old'),
    ('sweet', 'sour'),
    ('loud', 'quiet'),
    ('day', 'night'),
    ('easy', 'difficult'),
    ('near', 'far'),
    ('rich', 'poor'),
    ('smooth', 'rough'),
    ('thin', 'thick'),
    ('east', 'west'),
    ('yes', 'no'),
    ('male', 'female'),
    ('up', 'down'),
    ('left', 'right'),
    ('in', 'out'),
    ('begin', 'end'),
    ('before', 'after'),
    ('front', 'back'),
    ('more', 'less'),
    ('above', 'below'),
    ('push', 'pull'),
    ('enter', 'exit'),
    ('win', 'lose'),
    ('give', 'take'),
    ('black', 'white'),
    ('buy', 'sell'),
    ('add', 'subtract'),
    ('north', 'south'),
    ('under', 'over'),
    ('good', 'bad'),
    ('summer', 'winter'),
    ('wet', 'dry'),
    ('alive', 'dead'),
    ('first', 'last'),
    ('accept', 'reject'),
    ('always', 'never'),
    ('come', 'go'),
    ('laugh', 'cry'),
    ('single', 'married')
]


left = []
right = []
for pair in pairs:
    l = tokenize(pair[0])
    r = tokenize(pair[1])
    if len(l) > 2 or len(r) > 1:
        print(f"{pair} has too many tokens, skipping.")
        continue
    left.append(pair[0])
    right.append(pair[1])
word_list = left + right
n_words = len(word_list)
print(n_words)

# %%
for t in tokenizer.encode(' ' + ' '.join(word_list)):
    print(repr(tokenizer.decode(t)))

# %%
# left = "true hot tall big open happy light fast strong hard early full high inside clean new sweet loud day easy near rich smooth thin east yes male".split(' ')
# right = "false cold short small closed sad dark slow weak soft late empty low outside dirty old sour quiet night difficult far poor rough thick west no female".split(' ')

# %%

# for s in left + right:
#     if len(tokenize(s)) > 1:
#            print(s, tokenize(s))

# %%
# word_list = left + right
# n_words = len(word_list)
# %%
def embed(token_ids):
    x = model.W_E[token_ids]
    # TODO: gould seems to just take mlp0_out?
    x = x + model.blocks[0](x[:, None, :])[:, 0, :]
    # x = x + model.blocks[1](x[:, None, :])[:, 0, :]
    # x = x + model.blocks[2](x[:, None, :])[:, 0, :]
    return x

def unembed(x):
    return x @ model.W_U + model.b_U

def idx_to_head_no(idx):
    layer_no = idx // model.cfg.n_heads + 1
    head_no = idx % model.cfg.n_heads
    return layer_no, head_no

# %%
from transformer_lens import FactoredMatrix


def compute_OV(layer_no, head_no):
    tokens = torch.tensor(list(map(tokenize, word_list)))[:, 0]
    embeddings = embed(tokens)
    OV = model.blocks[layer_no].attn.OV[head_no]
    x = embeddings @ OV
    logits = unembed(x.AB)
    # probs = torch.log_softmax(logits, 1)
    token_logits: FactoredMatrix = logits[:, tokens]
    # token_probs = probs[:, tokens]
    return token_logits
    # return token_probs

def plot_OV(token_logits, layer_no, head_no):
    fig = px.imshow(
        token_logits,
        title=f"L{layer_no}H{head_no}",
        x=word_list,
        y=word_list,
        labels={
            'x': "logit",
            'y': "token",
        },
        color_continuous_scale='rdbu',
        color_continuous_midpoint=0,
        
    )
    fig.show()

# %%
# OVs = []
# alignments = []

# for layer_no in range(1, model.cfg.n_layers):
#     for head_no in range(model.cfg.n_heads):
#         my_OV = compute_OV(layer_no, head_no)
#         OVs.append(my_OV)
#         alignment = (my_OV.cpu().numpy().argmax(1) == list(range(24, 48)) + list(range(24))).sum()
#         alignments.append(alignment)

from tqdm import tqdm

OVs = []
alignments = []

total_iterations = (model.cfg.n_layers - 1) * model.cfg.n_heads
with tqdm(total=total_iterations, desc="Processing") as pbar:
    for layer_no in range(1, model.cfg.n_layers):
        for head_no in range(model.cfg.n_heads):
            my_OV = compute_OV(layer_no, head_no)
            OVs.append(my_OV)
            alignment = (my_OV.to(torch.float32).cpu().numpy().argmax(1) == list(range(n_words//2, n_words)) + list(range(n_words//2))).sum()
            alignments.append(alignment)
            pbar.update(1)

# %%
import numpy as np

alignments = np.stack(alignments)
OV_stack = torch.stack(OVs)
OV_stack.shape

# %%
for idx in np.argsort(-alignments)[:8]:
    print(alignments[idx])
    plot_OV(OV_stack[idx].to(torch.float32).cpu().numpy(), 1+idx//model.cfg.n_heads, idx%model.cfg.n_heads)

# %%
ov_idx = (14 - 1) * 16 + 12
plot_OV(OV_stack[ov_idx].to(torch.float32).cpu().numpy(), 1+ov_idx//model.cfg.n_heads, ov_idx%model.cfg.n_heads)
print(alignments[ov_idx])
# %%
# assert False

# %%
import numpy as np
experiment_word_list = []
experiment_string = ""
for l, r in zip(left, right):
    experiment_word_list.append(l)
    experiment_word_list.append(r)
    experiment_string += (f" {l} {r}\n")
# experiment_sequence = tokenizer.encode(' ' + ' '.join(experiment_word_list))
experiment_sequence = tokenizer.encode(experiment_string)

experiment_sequence = torch.tensor(experiment_sequence)
for t in experiment_sequence:
    print(repr(tokenizer.decode(t)))


# %%
from transformer_lens.utils import get_act_name
batch_size = 16

loss_diffs = {}

for block_no in range(1, model.cfg.n_layers):
# for block_no in [9]:
    for head_no in range(model.cfg.n_heads):
    # for head_no in [1]:

        # compute z_mean
        test_sequences = []
        for _ in range(batch_size):
            shuffled_word_list = [w for w in word_list]
            np.random.shuffle(shuffled_word_list)

            test_sequences.append(
                tokenizer.encode(' ' + ' '.join(shuffled_word_list))
            )

        test_batch = np.stack(test_sequences)
        test_batch = torch.tensor(test_batch).to(device)

        z_stack = []

        for sample in test_batch:
            _, cache = model.run_with_cache(sample[None, :])
            z = cache[f'blocks.{block_no}.attn.hook_z'][0, :, head_no, :]
            z_stack.append(z)

        z_stack = torch.stack(z_stack)
        z_mean = z_stack.mean((0, 1))

        def mean_ablation_hook(value, hook):
            value[:, :, head_no, :] = z_mean
            return value

        original_loss = model(
            experiment_sequence,
            return_type="loss",
            loss_per_token=True
        )

        ablated_loss = model.run_with_hooks(
            experiment_sequence, 
            return_type="loss",
            loss_per_token=True,
            fwd_hooks=[(
                get_act_name("z", block_no), 
                mean_ablation_hook
                )]
            )

        loss_difference = ablated_loss[:, 1::3].mean() - original_loss[:, 1::3].mean()
        head_id = f"L{block_no}H{head_no}"
        loss_diffs[head_id] = loss_difference
        print(f"Loss difference for {head_id}: {loss_difference.item():.3f}")

{k: v for k, v in sorted(loss_diffs.items(), key=lambda item: item[1], reverse=True)}

# %%
import pandas as pd
x = []
y = []
head_ids = []

for idx, (loss_diff, alignment) in enumerate(zip(loss_diffs.values(), alignments)):
    loss_diff = loss_diff.item()
    x.append(alignment)
    y.append(loss_diff)
    layer_no, head_no = idx_to_head_no(idx)
    head_ids.append(f"L{layer_no}H{head_no}")
    # print(f"L{layer_no}H{head_no}:\t{loss_diff:.3f}\t{alignment}")

df = pd.DataFrame({
    "tally_score": x,
    "ablation_score": y,
    "head_id": head_ids,
})

fig = px.scatter(
    df,
    x="tally_score",
    y="ablation_score",
    hover_data=["head_id"],
    labels={
        'tally_score': 'Tally Score',
        'ablation_score': 'Ablation Effect (loss difference)'
    },
    title="Ablation Effect vs Tally Scores in Pythia-1.4b"
)
fig.show()

# %%
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(df['tally_score'], df['ablation_score'])
r_squared = r_value**2
print(f"R-squared: {r_squared:.4f}")
    
# print(f"Original Loss: {original_loss.item():.3f}")
# print(f"Ablated Loss: {ablated_loss.item():.3f}")
# %%
print(z_mean)
print(get_act_name('z', block_no))

# %%
from sklearn.decomposition import FastICA

# n_components = OV_stack.shape[0]
OV_stack_np = OV_stack.cpu().numpy()
OV_stack_np = OV_stack_np.reshape(OV_stack_np.shape[0], -1)
print(OV_stack_np.shape)

n_components = (model.cfg.n_layers - 1) * model.cfg.n_heads - 1

ica = FastICA(n_components=n_components, random_state=42, max_iter=10000)
components = ica.fit_transform(OV_stack_np)

# %%
px.imshow(components[:100, :100])  # should be sparse
# %%
import numpy as np

composite = OV_stack_np.T @ components
composite.shape
composite = composite.reshape((n_words, n_words, n_components)).transpose((2, 0 ,1))

composite_alignments = []
for i, mat in enumerate(composite):
    alignment = (mat.argmax(1) == list(range(n_words//2, n_words)) + list(range(n_words//2))).sum()
    composite_alignments.append(alignment)
   
# %%
component_alignments = np.stack(composite_alignments)

sorted_alignments = np.argsort(-component_alignments)

# %%
for idx in sorted_alignments[:6]:
    print(component_alignments[idx])
    plot_OV(composite[idx], idx, '')

# %%
# Examine head coefficients for top component
top_component_id = sorted_alignments[0]
component_head_ids = np.argwhere(np.abs(components[:, top_component_id]) > 0.1)[:, 0]


# %%
for head_id in component_head_ids:
    layer_no = head_id // model.cfg.n_heads + 1
    head_no = head_id % model.cfg.n_heads
    print(f"L{layer_no}H{head_no}: {components[head_id, top_component_id]}")
    plot_OV(OV_stack[head_id].cpu().numpy(), layer_no, head_no)


# %%
