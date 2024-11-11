# %%

import os

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from tqdm import tqdm
from transformer_lens import FactoredMatrix, HookedTransformer

torch.set_grad_enabled(False)

device = 'cuda'

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

def compute_OV(model, layer_no, head_no):
    tokenize = lambda x: model.tokenizer.encode(f" {x}", add_special_tokens=False)
    tokens = torch.tensor(list(map(tokenize, word_list)))[:, 0]
    embeddings = embed(model, tokens)
    OV = model.blocks[layer_no].attn.OV[head_no]
    x = embeddings @ OV
    logits = unembed(model, x.AB)
    token_logits: FactoredMatrix = logits[:, tokens]
    return token_logits

def plot_OV(token_logits, title=''):
    fig = px.imshow(
        token_logits,
        title=title,
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

def convert_to_polar(eigenvalues):
    real_parts = eigenvalues.real
    imag_parts = eigenvalues.imag

    magnitude = torch.sqrt(real_parts**2 + imag_parts**2)
    log_magnitude = torch.log(magnitude)

    theta = torch.atan2(imag_parts, real_parts)
    theta = torch.rad2deg(theta)
    return log_magnitude, theta


def plot_polar_eigenvalues(L, title=None):
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
        title=title,
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
from tqdm import tqdm


def compute_OV_and_tally_scores_for_head(model, layer_no, head_no):
    my_OV = compute_OV(model, layer_no, head_no)
    alignment = (my_OV.to(torch.float32).cpu().numpy().argmax(1) == list(range(n_words//2, n_words)) + list(range(n_words//2))).sum()
    return my_OV, alignment


def compute_OV_and_tally_scores_for_model(model):
    OVs = []
    alignments = []

    total_iterations = (model.cfg.n_layers - 1) * model.cfg.n_heads
    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for layer_no in range(1, model.cfg.n_layers):
            for head_no in range(model.cfg.n_heads):
                my_OV, alignment = compute_OV_and_tally_scores_for_head(model, layer_no, head_no)
                OVs.append(my_OV)
                alignments.append(alignment)
                pbar.update(1)
    return OVs, alignments


# %%
import numpy as np

model_names = {
    "gpt2-small": 137,
    "gpt2-medium": 380,
    "gpt2-large": 812,
    "gpt2-xl": 1610,
    "EleutherAI/pythia-70m": 96,
    "EleutherAI/pythia-160m": 213,
    "EleutherAI/pythia-410m": 506,
    "EleutherAI/pythia-1b": 1080,
    "EleutherAI/pythia-1.4b": 1520,
    "EleutherAI/pythia-2.8b": 2910,
    "meta-llama/Llama-3.2-1B": 1240,
    "meta-llama/Llama-3.2-3B": 3210,
    "google/gemma-2-2b": 2610,
}

top_heads = {
    "google/gemma-2-2b": (22, 0),
    "meta-llama/Llama-3.2-3B": (19, 18),
    'meta-llama/Llama-3.2-1B': (11, 23),
    'EleutherAI/pythia-2.8b': (15, 3),
    'EleutherAI/pythia-1.4b': (13, 2),
    'EleutherAI/pythia-1b': (9, 1),
    'EleutherAI/pythia-410m': (13, 1),
    'EleutherAI/pythia-160m': (5, 9),
    'EleutherAI/pythia-70m': (3, 1),
    'gpt2-xl': (26, 10),
    'gpt2-large': (25, 5),
    'gpt2-medium': (13, 2),
    'gpt2-small': (9, 7)
}

tally_scores = {
    "gpt2-small": 28,
    "gpt2-medium": 34,
    "gpt2-large": 60,
    "gpt2-xl": 52,
    "EleutherAI/pythia-70m": 8,
    "EleutherAI/pythia-160m": 19,
    "EleutherAI/pythia-410m": 29,
    "EleutherAI/pythia-1b": 67,
    "EleutherAI/pythia-1.4b": 70,
    "EleutherAI/pythia-2.8b": 28,
    "meta-llama/Llama-3.2-1B": 31,
    "meta-llama/Llama-3.2-3B": 42,
    "google/gemma-2-2b": 72,
}


# %%
tally_scores = {n: None for n in model_names}
top_ovs = {n: None for n in model_names}

# %%
import gc

for i, model_name in enumerate(model_names):
    try:
        del(my_model)
        gc.collect()
        torch.cuda.empty_cache()
    except NameError:
        pass
    print(model_name)
    my_model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=True,
        device=device
    )
    block_no, head_no = top_heads[model_name]

    token_OV, alignment = compute_OV_and_tally_scores_for_head(my_model, block_no, head_no)

    print(alignment)
    OV_fig = plot_OV(token_OV.to(torch.float32).cpu().numpy(), f"{model_name}: L{block_no}H{head_no}")
    OV_fig.update_layout(title_x=0.5)
    OV_fig.update_layout(margin=dict(l=0, r=0, t=80, b=40))

    OV_fig.update_layout(
        autosize=False,
        width=440,
        height=440
    )
    OV_fig.show()
    OV_fig.write_image(
        # f"OV_plots/{i}_OV_{model_name}_L{block_no}H{head_no}.png",
        f"OV_plots/plot_{i}.png",
        scale=2,
    )
    OV = my_model.blocks[block_no].attn.OV[head_no]
    L, V = torch.linalg.eig(OV.AB.T.cpu())
    filtered_L = L[L.real.abs() > 1e-3]
    eigen_fig = plot_polar_eigenvalues(filtered_L, f"{model_name}: L{block_no}H{head_no}")
    eigen_fig.update_layout(title_x=0.5)
    eigen_fig.update_layout(margin=dict(l=0, r=0, t=80, b=40))
    eigen_fig.update_layout(
        autosize=False,
        width=440,
        height=440
    )
    eigen_fig.show()
    eigen_fig.write_image(
        f"eigen_plots/plot_{i}.png",
        scale=2
    )
    tally_scores[model_name] = alignment

    del(my_model)
    gc.collect()
    torch.cuda.empty_cache()


import gc

import pandas as pd
# %%
import plotly.express as px

data = {
    "model_name": list(model_names.keys()),
    "model_size": list(model_names.values()),
    "tally_score": [tally_scores[name] / 108 for name in model_names]
}

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

df = pd.DataFrame(data)

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

fig.update_layout(
    autosize=False,
    width=660,
    height=440
)

fig.show()

fig.write_image(
    "tally_scores.png",
    scale=2
)


# ablation study

# %%
from transformer_lens.utils import get_act_name

experiment_word_list = []
experiment_string = ""
for l, r in zip(left, right):
    experiment_word_list.append(l)
    experiment_word_list.append(r)
    experiment_string += (f" {l} {r}\n")

batch_size = 16
def compute_ablation_effect(model):
    loss_diffs = {}
    tokenizer = model.tokenizer
    experiment_sequence = tokenizer.encode(experiment_string)
    if experiment_sequence[0] != tokenizer.bos_token_id:
        print("adding bos token")
        experiment_sequence = [tokenizer.bos_token_id] + experiment_sequence

    experiment_sequence = torch.tensor(experiment_sequence)

    pbar = tqdm(total=model.cfg.n_layers * model.cfg.n_heads)
    for block_no in range(1, model.cfg.n_layers):
        for head_no in range(model.cfg.n_heads):
            pbar.update(1)

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
            # print(f"Loss difference for {head_id}: {loss_difference.item():.3f}")

    print({k: v for k, v in sorted(loss_diffs.items(), key=lambda item: item[1], reverse=True)})
    return loss_diffs


import pandas as pd
from scipy.stats import linregress

for model_name in [
    # "EleutherAI/pythia-1b",
    # "EleutherAI/pythia-1.4b",
    # "google/gemma-2-2b",
    "gpt2-large"
]:
    try:
        del(my_model)
        gc.collect()
        torch.cuda.empty_cache()
    except NameError:
        pass
    print(model_name)

    my_model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=True,
        device=device
    )

    OVs, alignments = compute_OV_and_tally_scores_for_model(my_model)

    my_loss_diffs = compute_ablation_effect(my_model)

    x = []
    y = []
    head_ids = []

    for idx, (loss_diff, alignment) in enumerate(zip(my_loss_diffs.values(), alignments)):
        loss_diff = loss_diff.item()
        x.append(alignment)
        y.append(loss_diff)
        layer_no, head_no = idx_to_head_no(my_model, idx)
        head_ids.append(f"L{layer_no}H{head_no}")

    top_block, top_head = top_heads[model_name]
    hoi = f"L{top_block}H{top_head}"

    df = pd.DataFrame({
        "tally_score": [s/108 for s in x],
        "ablation_score": y,
        "head_id": head_ids,
        "is_hoi": [head_id == hoi for head_id in head_ids],
    })

    slope, intercept, r_value, p_value, std_err = linregress(df['tally_score'], df['ablation_score'])
    r_squared = r_value**2
    print(f"R-squared: {r_squared:.4f}")

    fig = px.scatter(
        df,
        x="tally_score",
        y="ablation_score",
        color="is_hoi",
        # set color legend labels
        hover_data=["head_id"],
        labels={
            'tally_score': 'Tally Score',
            'ablation_score': 'Ablation Effect (loss difference)'
        },
        title=f"Ablation Effect vs Tally Scores<br>{model_name}"
    )
    fig.add_annotation(
      x=0.97, y=0.97, xref="paper", yref="paper",
      text=f"R^2: {r_squared:.2f}",
      showarrow=False,
      font=dict(size=12)
    )
    y_max = df['ablation_score'].max()
    y_buffer = y_max * 0.15
    fig.update_layout(
        yaxis=dict(
            range=[None, y_max + y_buffer]
        )
    )

    fig.update_layout(showlegend=False)
    fig.update_layout(
        autosize=False,
        width=440,
        height=440,
    )
    fig.show()
    fig.write_image(
        f"ablation_plots/{model_name.split('/')[-1]}.png",
        scale=2
    )

    # break