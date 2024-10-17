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

model = HookedTransformer.from_pretrained(
    "gpt2-large",
    fold_ln=True,
)

# %%
tokenizer = model.tokenizer

# %%
import pandas as pd
import plotly.express as px

# %%
left = "true hot tall big open happy light fast strong hard early full high inside clean new sweet loud day easy near rich smooth thin".split(' ')
right = "false cold short small closed sad dark slow weak soft late empty low outside dirty old sour quiet night difficult far poor rough thick".split(' ')

list(zip(left, right))

# %%

def tokenize(str):
    return tokenizer(f" {str}").input_ids

for s in left + right:
    if len(tokenize(s)) > 1:
           print(s, tokenize(s))

# %%
word_list = left + right
# %%
def embed(token_ids):
    x = model.W_E[token_ids]
    # TODO: gould seems to just take mlp0_out?
    x = model.blocks[0](x[:, None, :])[:, 0, :]
    return x

def unembed(x):
    return x @ model.W_U + model.b_U

# %%
from transformer_lens import FactoredMatrix


def compute_OV(layer_no, head_no):
    tokens = torch.tensor(list(map(tokenize, word_list)))[:, 0]
    embeddings = embed(tokens)
    OV = model.blocks[layer_no].attn.OV[head_no]
    x = embeddings @ OV
    logits = unembed(x.AB)
    print(logits.shape)
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
OVs = []
alignments = []

for layer_no in range(1, model.cfg.n_layers):
    for head_no in range(model.cfg.n_heads):
        my_OV = compute_OV(layer_no, head_no)
        OVs.append(my_OV)
        alignment = (my_OV.cpu().numpy().argmax(1) == list(range(24, 48)) + list(range(24))).sum()
        alignments.append(alignment)

# %%
import numpy as np

alignments = np.stack(alignments)
OV_stack = torch.stack(OVs)
OV_stack.shape

# %%
for idx in np.argsort(-alignments)[:16]:
    print(alignments[idx])
    plot_OV(OV_stack[idx].cpu().numpy(), 1+idx//model.cfg.n_heads, idx%model.cfg.n_heads)

# %%
from sklearn.decomposition import FastICA

# n_components = OV_stack.shape[0]
OV_stack_np = OV_stack.cpu().numpy()
OV_stack_np = OV_stack_np.reshape(OV_stack_np.shape[0], -1)
print(OV_stack_np.shape)

n_components = (model.cfg.n_layers - 1) * model.cfg.n_heads - 1

ica = FastICA(n_components=n_components, random_state=42, max_iter=10000)
components = ica.fit_transform(OV_stack_np)

components.shape

# %%
components
# %%
px.imshow(components)
# %%
import numpy as np

composite = OV_stack_np.T @ components
composite.shape
composite = composite.reshape((48, 48, n_components)).transpose((2, 0 ,1))


composite_alignments = []
for i, mat in enumerate(composite):
    alignment = (mat.argmax(1) == list(range(24, 48)) + list(range(24))).sum()
    composite_alignments.append(alignment)
   
# %%
component_alignments = np.stack(composite_alignments)

for idx in np.argsort(-component_alignments)[:16]:
    print(component_alignments[idx])
    plot_OV(composite[idx], idx, '')



# medium:
# 79 is vibe similarity
# 180 looks anyonym-ish
# 261 is also vibe similarity
# %%
OV_stack_np.shape
# %%
components[:, 0].shape
# 26 looks like a succession component
# %%

