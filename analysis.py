# %%

# Standard imports
import os
import torch
from tqdm import tqdm
import plotly.express as px

os.environ["HF_TOKEN"] = "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"

torch.set_grad_enabled(False)

# %%

from transformer_lens import HookedTransformer

device = 'mps'

# %%

my_model = HookedTransformer.from_pretrained(
    # "gpt2-xl",
    "EleutherAI/pythia-1b",
    # "meta-llama/Llama-3.2-1B",
    fold_ln=True,
    device=device
)

# %%
my_model = my_model.to(torch.bfloat16)

# %%
tokenizer = my_model.tokenizer

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
    left.append(pair[0])
    right.append(pair[1])
word_list = left + right
n_words = len(word_list)
print(n_words)

# %%
left = []
right = []
for pair in pairs:
    l = tokenize(pair[0])
    r = tokenize(pair[1])
    if len(l) > 2 or len(r) > 1:
        print(f"{pair} has too many tokens, skipping.")
        continue


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
def embed(model, token_ids):
    x = model.W_E[token_ids]
    # TODO: gould seems to just take mlp0_out?
    x = model.blocks[0](x[:, None, :])[:, 0, :]
    return x

def unembed(model, x):
    return x @ model.W_U + model.b_U

def idx_to_head_no(model, idx):
    layer_no = idx // model.cfg.n_heads + 1
    head_no = idx % model.cfg.n_heads
    return layer_no, head_no

# %%
from transformer_lens import FactoredMatrix


def compute_OV(model, layer_no, head_no):
    tokenize = lambda x: model.tokenizer.encode(f" {x}", add_special_tokens=False)
    tokens = torch.tensor(list(map(tokenize, word_list)))[:, 0]
    embeddings = embed(model, tokens)
    OV = model.blocks[layer_no].attn.OV[head_no]
    x = embeddings @ OV
    logits = unembed(model, x.AB)
    # probs = torch.log_softmax(logits, 1)
    token_logits: FactoredMatrix = logits[:, tokens]
    # token_probs = probs[:, tokens]
    return token_logits
    # return token_probs

def plot_OV(token_logits, layer_no, head_no):
    fig = px.imshow(
        token_logits,
        title=f"L{layer_no}H{head_no}",
        x=word_list[:len(token_logits)],
        y=word_list[:len(token_logits)],
        labels={
            'x': "logit",
            'y': "token",
        },
        color_continuous_scale='rdbu',
        color_continuous_midpoint=0,
        
    )
    return fig
    # fig.show()

# import plotly.graph_objects as go

# def plot_OV(token_logits, layer_no, head_no):
#     fig = go.Figure(
#         data=go.Heatmap(
#             z=token_logits[::-1, :],
#             x=word_list[:len(token_logits)],
#             y=word_list[:len(token_logits)][::-1],
#             colorscale='RdBu',
#             zmid=0  # Midpoint for the diverging color scale
#         )
#     )

#     fig.update_layout(
#         title=f"L{layer_no}H{head_no}",
#         xaxis_title="logit",
#         yaxis_title="token"
#     )

#     fig.update_xaxes(scaleanchor="y", scaleratio=1)
#     fig.update_yaxes(scaleanchor="x", scaleratio=1)

#     return fig
    # fig.show()

def convert_to_polar(eigenvalues):
    real_parts = eigenvalues.real
    imag_parts = eigenvalues.imag

    # Compute the magnitude (r) and then take the log of the magnitude
    magnitude = torch.sqrt(real_parts**2 + imag_parts**2)
    log_magnitude = torch.log(magnitude)  # or torch.log10(magnitude) for base-10

    # Compute the angle (theta)
    theta = torch.atan2(imag_parts, real_parts)
    theta = torch.rad2deg(theta)
    return log_magnitude, theta

import pandas as pd

def plot_polar_eigenvalues(L):
    log_magnitude, theta = convert_to_polar(L)
    df = pd.DataFrame({
        'log_magnitude': log_magnitude.numpy(),
        'theta': theta.numpy(),
        'idx': range(len(L))
    })

    fig = px.scatter_polar(
        df,
        r='log_magnitude',
        theta='theta', 
        title='Eigenvalues in Polar Coordinates',
        labels={
            'log_magnitude': 'Log Magnitude',
            'theta': 'Angle (degrees)',
            'idx': 'idx',
        },
        hover_data={'idx': True},
        start_angle=0,
    )

    return fig
    # fig.show()


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

def compute_OV_and_tally_scores_for_model(model):
    OVs = []
    alignments = []

    total_iterations = (model.cfg.n_layers - 1) * model.cfg.n_heads
    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for layer_no in range(1, model.cfg.n_layers):
            for head_no in range(model.cfg.n_heads):
                my_OV = compute_OV(model, layer_no, head_no)
                OVs.append(my_OV)
                alignment = (my_OV.to(torch.float32).cpu().numpy().argmax(1) == list(range(n_words//2, n_words)) + list(range(n_words//2))).sum()
                alignments.append(alignment)
                pbar.update(1)
    return OVs, alignments

# %%
OVs, alignments = compute_OV_and_tally_scores_for_model(my_model)

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
    
# %%
import numpy as np

model_names = {
    # "gpt2": 137,
    # "gpt2-medium": 380,
    # "gpt2-large": 812,
    # "gpt2-xl": 1610,
    # "EleutherAI/pythia-70m": 96,
    # "EleutherAI/pythia-160m": 213,
    # "EleutherAI/pythia-410m": 506,
    # "EleutherAI/pythia-1b": 1080,
    # "EleutherAI/pythia-1.4b": 1520,
    # "EleutherAI/pythia-2.8b": 2910,
    # "meta-llama/Llama-3.2-1B": 1240,
    # "meta-llama/Llama-3.2-3B": 3210,
    "google/gemma-2-2b": 2610,
}

model_names = dict(reversed(model_names.items()))

tally_scores = {n: None for n in model_names}
top_ovs = {n: None for n in model_names}
top_heads = {n: None for n in model_names}

# %%
import gc

for model_name in model_names:
    print(model_name)
    my_model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=True,
        device=device
    )

    # OVs, alignments = compute_OV_and_tally_scores_for_model(my_model)

    # alignments = np.stack(alignments)
    # OV_stack = torch.stack(OVs)

    top_idx = None
    for idx in np.argsort(-alignments)[:1]:
        if top_idx is None:
            top_idx = idx
        print(alignments[idx])
        block_no = 1+idx//my_model.cfg.n_heads
        head_no = idx%my_model.cfg.n_heads
        plot_OV(OV_stack[idx].to(torch.float32).cpu().numpy(), block_no, head_no).show()
        OV = my_model.blocks[block_no].attn.OV[head_no]
        L, V = torch.linalg.eig(OV.AB.T.cpu())
        filtered_L = L[L.real.abs() > 1e-3]
        plot_polar_eigenvalues(filtered_L).show()

    tally_scores[model_name] = alignments[top_idx].item()
    top_ovs[model_name] = OVs[top_idx].cpu().numpy()
    block_no, head_no = idx_to_head_no(my_model, top_idx)
    top_heads[model_name] = (block_no.item(), head_no.item())
    del(my_model)
    gc.collect()

# %%
import plotly.express as px
import pandas as pd

# Data preparation
data = {
    "model_name": list(model_names.keys()),
    "model_size": list(model_names.values()),
    "tally_score": [tally_scores[name] for name in model_names]
}

# Identify model family
def get_family(name):
    if "gpt" in name:
        return "GPT2"
    elif "pythia" in name:
        return "Pythia"
    elif "llama" in name.lower():
        return "Llama 3.2"
    elif "gemma" in name:
        return "Gemma 2"
    else:
        return "Other"

data["model_family"] = [get_family(name) for name in data["model_name"]]

# Convert to DataFrame
df = pd.DataFrame(data)

# Create the line plot
fig = px.line(
    df,
    x="model_size",
    y="tally_score",
    log_x=True,
    color="model_family",
    markers=True,
    labels={"model_size": "Model Size (millions of parameters)", "tally_score": "Tally Scores", "model_family": "Model Family"},
    title="Max Tally Scores by Model Size"
)

# Show the plot
fig.show()

# %%
# save computed dictionaries
import pickle

with open("tally_scores.pkl", "wb") as f:
    pickle.dump(tally_scores, f)

with open("top_ovs.pkl", "wb") as f:
    pickle.dump(top_ovs, f)

with open("top_heads.pkl", "wb") as f:
    pickle.dump(top_heads, f)

# %%
import gc
del(my_model)
gc.collect()
my_model = HookedTransformer.from_pretrained(
    # "gpt2-xl",
    # "EleutherAI/pythia-1b",
    # "meta-llama/Llama-3.2-3B",
    "google/gemma-2-2b",
    fold_ln=True,
    device=device
)

import numpy as np

# %%
OVs, alignments = compute_OV_and_tally_scores_for_model(my_model)

alignments = np.stack(alignments)
OV_stack = torch.stack(OVs)

# %%


# %%
from plotly.subplots import make_subplots

n_row = 1
n_col = 1

# set large width and height
fig = make_subplots(rows=n_row, cols=n_col*2,
    specs=[[{"type": "polar"}, {"type": "xy"}] * n_col] * n_row,
    subplot_titles=[f"Head {i}" for i in range(1, 9)],
)
fig.update_layout(
    autosize=False,
    width=400*n_col*2,
    height=400*n_row,
    margin=dict(l=80, r=80, t=80, b=0),
)

for i, idx in enumerate(np.argsort(-alignments)[:1]):
    block_no, head_no = 1+idx//my_model.cfg.n_heads, idx%my_model.cfg.n_heads
    print(f"L{block_no}H{head_no}")
    print("tally score:", alignments[idx])
    OV_plot = plot_OV(OV_stack[idx].to(torch.float32).cpu().numpy(), 1+idx//my_model.cfg.n_heads, idx%my_model.cfg.n_heads)
    # OV_plot.show()
    # break
    OV = my_model.blocks[block_no].attn.OV[head_no]
    L, V = torch.linalg.eig(OV.AB.T.cpu())
    # filter out L under 1e-8
    filtered_L = L[L.real.abs() > 1e-3]
    filtered_V = V[:, L.real.abs() > 1e-3]
    print(filtered_L.shape)
    print(filtered_V.shape)
    eigen_plot = plot_polar_eigenvalues(filtered_L)
    # print proportion of positive eigenvalues:
    print((filtered_L .real > 0).sum().item() / len(filtered_L))
    row_no = 1+i//n_col
    col_no = 1+(i%n_col)*2
    for trace in OV_plot.data:
        fig.add_trace(trace, row=row_no, col=col_no+1)
    for trace in eigen_plot.data:
        fig.add_trace(trace, row=row_no, col=col_no)
    # set automargins
    fig.update_xaxes(scaleanchor='y', scaleratio=1, row=row_no, col=col_no+1)
    # fig.update_yaxes(scaleanchor='x', scaleratio=1, row=row_no, col=col_no+1)
fig.show()

# %%
# fig.show()

 # %%
def get_projection_magnitude(a, b):
    return torch.dot(a, b) / b.norm()

# W_U_pinv = torch.linalg.pinv(my_model.W_U)

def reverse_unembed(model, token_id):
    one_hot = torch.nn.functional.one_hot(token_id, num_classes=model.W_U.shape[1])[0]
    print(one_hot.shape)
    pre_bias_logits = one_hot - model.b_U
    return pre_bias_logits @ W_U_pinv

# %%
print(token_b)
embedding_b = reverse_unembed(my_model, token_b)
my_model.unembed(embedding_b.to(device))

# %%
one_hot = torch.nn.functional.one_hot(token_b, num_classes=my_model.W_U.shape[1])[0].float()
print(one_hot)
print(one_hot.max())
temp = one_hot @ W_U_pinv @ my_model.W_U
temp.max()

# %%
print("W_U shape:", my_model.W_U.shape)
print("W_U_pinv shape:", W_U_pinv.shape)
print("One-hot vector shape:", one_hot.shape)

# %%
word_a = "rough"
word_b = "hard"
token_a = my_model.tokenizer.encode(f" {word_a}", add_special_tokens=False, return_tensors='pt')[0].to(device)
token_b = my_model.tokenizer.encode(f" {word_b}", add_special_tokens=False, return_tensors='pt')[0].to(device)

embedding_a = embed(my_model, token_a).cpu()
embedding_b = embed(my_model, [token_b]).cpu()
# embedding_b = reverse_unembed(my_model, token_b).cpu()

print(unembed(my_model, embedding_b.to(device)).max())

# eigen_idxs = torch.arange(10)
# eigen_idxs = torch.tensor([0, 2, 3, 5, 6, 8, 9])
# eigen_idxs = torch.randperm(255)[:7]
eigen_idxs = torch.arange(255)

ratios = []
for i, eigenvector in enumerate(filtered_V[:, eigen_idxs].T):
    proj_a = get_projection_magnitude(embedding_a[0], eigenvector.real)
    proj_b = get_projection_magnitude(embedding_b[0], eigenvector.real)
    ratio = proj_a / proj_b
    ratios.append(ratio)

ratios = torch.stack(ratios)
negative_ratio_proportion = (ratios < 0).sum().item() / len(ratios)
print(negative_ratio_proportion)
    # print(f"Eigenvalue {eigen_idxs[i]}: {proj_a:.3f} {proj_b:.3f}\tratio: {proj_a / proj_b:.3f}")
# %%
filtered_V.shape

# %%
# get average of first 10 eigenvectors
interesting_vector = filtered_V[:, :10].mean(1)
interesting_unit_vector = interesting_vector.real / torch.linalg.norm(interesting_vector.real)
interesting_unit_vector.norm()



# interesting_vector = filtered_V[:, 0]
# interesting_unit_vector = interesting_vector.real / torch.linalg.norm(interesting_vector.real)

# %%
def negate_direction(vec, d):
    projection = torch.dot(vec, d) * d
    vec_new = vec - 2 * projection
    return vec_new

# %%

# def approx_transform(vec, eigenvectors, eigenvalues, selected_indices):
#     return sum((torch.dot(vec, eigenvectors[:, i].real) * eigenvalues[i].real * eigenvectors[:, i].real) 
#                for i in selected_indices)

def approx_transform(vec, eigenvectors, eigenvalues, selected_indices):
    vec = vec.to(dtype=torch.complex64)
    return sum((torch.dot(vec, eigenvectors[:, i]) * eigenvalues[i] * eigenvectors[:, i])
               for i in selected_indices)

def approx_transform_2(vec, V, L, _=None):
    vec = vec.to(dtype=torch.complex64).cpu()
    V = V.cpu()
    L = L.cpu()
    L[:10] = 0.
    vec = vec.to(dtype=torch.complex64, device='cpu')
    # Solve V @ x = vec for x
    V_inv_vec = torch.linalg.solve(V, vec)
    # Apply the diagonal eigenvalue matrix
    transformed = V @ (L * V_inv_vec)
    return transformed  # Convert back to real if O is real

import torch

def approx_transform_3(vec, V, L, selected_indices):
    vec = vec.to(dtype=torch.complex64).cpu()
    L = L.cpu()
    V = V.cpu()

    # Select the desired eigenvalues and eigenvectors
    L_selected = L[selected_indices]
    V_selected = V[:, selected_indices]

    # Compute the coefficients c
    c = vec @ V_selected  # (1 x m) @ (m x k) -> (1 x k)

    # Apply the selected eigenvalues
    c_transformed = c * L_selected  # Element-wise multiplication

    # Reconstruct the transformed vector
    transformed_vec = c_transformed @ V_selected.conj().T  # (1 x k) @ (k x m) -> (1 x m)

    # If the result should be real, take the real part
    transformed_vec = transformed_vec

    return transformed_vec.squeeze()


# %%
for w in word_list:
    token = my_model.tokenizer.encode(f" {w}", add_special_tokens=False)[0]
    embedding = embed(my_model, [token])
    # opposite_embedding = negate_direction(embedding[0], interesting_unit_vector.to(device))
    opposite_embedding = embedding @ OV.AB
    # logits = unembed(my_model, embedding)
    logits = embedding @ my_model.W_E.T
    predicted_token = torch.argmax(logits).item()
    opposite_logits = opposite_embedding @ my_model.W_E.T
    opposite_predicted_token = torch.argmax(opposite_logits).item()
    print(
        w,
        my_model.tokenizer.decode([predicted_token]),
        my_model.tokenizer.decode([opposite_predicted_token])
    )

# %%

# %%
selected_indices = torch.arange(2048)

tokenize = lambda x: my_model.tokenizer.encode(f" {x}", add_special_tokens=False)
tokens = torch.tensor(list(map(tokenize, word_list)))[:, 0]
embeddings = embed(my_model, tokens)
opposite_embeddings = []

L, V = torch.linalg.eig(OV.AB.T.cpu())

for e in tqdm(embeddings):
    # opposite_embedding = negate_direction(e, interesting_unit_vector.to(device))
    opposite_embedding = approx_transform_2(e, V.to(device), L.to(device), selected_indices)
    # opposite_embedding = e @ OV.AB
    opposite_embeddings.append(opposite_embedding)
opposite_embeddings = torch.stack(opposite_embeddings)
# assert opposite_embeddings.imag.abs().max() < 1e-2
opposite_embeddings = opposite_embeddings.real.to(device)
logits = unembed(my_model, opposite_embeddings)
token_logits = logits[:, tokens]

# %%

vals = token_logits.to(torch.float32).cpu().numpy()
vals

plot_OV(vals, '', '')

# %%
token_logits.argmax(1)

# %%

(token_logits.argmax(1).cpu().numpy() == list(range(n_words//2, n_words)) + list(range(n_words//2))).sum()

# %%
opposite_embeddings.imag.abs().max()

# %%
filtered_L = L[L.real.abs() > 1e-3]
plot_polar_eigenvalues(filtered_L)


# %%

    # x = embeddings @ OV
    # logits = unembed(model, x.AB)
    # # probs = torch.log_softmax(logits, 1)
    # token_logits: FactoredMatrix = logits[:, tokens]
    # # token_probs = probs[:, tokens]
    # return token_logits

# for idx in np.random.choice(len(alignments), 5):
#     block_no, head_no = 1+idx//my_model.cfg.n_heads, idx%my_model.cfg.n_heads
#     print(f"L{block_no}H{head_no}")
#     print("tally score:", alignments[idx])
#     plot_OV(OV_stack[idx].to(torch.float32).cpu().numpy(), 1+idx//my_model.cfg.n_heads, idx%my_model.cfg.n_heads)
#     OV = my_model.blocks[block_no].attn.OV[head_no]
#     L, V = torch.linalg.eig(OV.AB.cpu())
#     # filter out L under 1e-8
#     L = L[L.real.abs() > 1e-3]
#     print(L.shape)
#     plot_polar_eigenvalues(L)
#     # print proportion of positive eigenvalues:
#     print((L.real > 0).sum().item() / len(L))


# %%
assert False


# %%
len(alignments)




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

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Sample data
df = px.data.iris()

# Create individual figures
fig1 = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig2 = px.histogram(df, x="petal_length", color="species")

# Create subplot
fig = make_subplots(rows=1, cols=2)

# Add traces from each Plotly Express figure
for trace in fig1.data:
    fig.add_trace(trace, row=1, col=1)
for trace in fig2.data:
    fig.add_trace(trace, row=1, col=2)

fig.update_layout(title_text="Subplots with Plotly Express")
fig.show()

# %%
