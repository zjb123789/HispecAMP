#!/usr/bin/env python3
"""
Taxonomy-aware microbial representation pretraining
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import json
import numpy as np
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from safetensors.torch import save_file

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"[Reproducibility] SEED = {SEED}")

# =========================================================
# config
# =========================================================

JSON_PATH = "pretraining_data/lineage_ranked.json"
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
BATCH_SIZE = 32
EPOCHS = 15
LR = 5e-6
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./taxonomy_encoder"

# =========================================================
# load data
# =========================================================

print("Loading data...")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================================================
# samples
# =========================================================

samples = []

for taxid, info in data.items():

    r = info["ranks"]

    samples.append({
        "taxid": int(taxid),

        "text":
            f"[PHYLUM] {r['phylum']} "
            f"[CLASS] {r['class']} "
            f"[ORDER] {r['order']} "
            f"[FAMILY] {r['family']} "
            f"[GENUS] {r['genus']} "
            f"[SPECIES] {info['scientific_name']}",

        "species": info["scientific_name"],

        "phylum": r["phylum"],
        "class": r["class"],
        "order": r["order"],
        "family": r["family"],
        "genus": r["genus"],
    })

print("samples:", len(samples))

# =========================================================
# taxonomy maps
# =========================================================

genus_map = defaultdict(list)
family_map = defaultdict(list)
order_map = defaultdict(list)
class_map = defaultdict(list)
phylum_map = defaultdict(list)

for i, s in enumerate(samples):

    genus_map[s["genus"]].append(i)
    family_map[s["family"]].append(i)
    order_map[s["order"]].append(i)
    class_map[s["class"]].append(i)
    phylum_map[s["phylum"]].append(i)

all_indices = list(range(len(samples)))

# =========================================================
# sampler
# =========================================================

class TaxonomyBatchSampler:

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):

        order = []

        while len(order) < len(samples):

            batch = []

            g = random.choice(list(genus_map.keys()))
            if len(genus_map[g]) >= 2:
                batch += random.sample(genus_map[g], 2)

            f = random.choice(list(family_map.keys()))
            batch.append(random.choice(family_map[f]))

            o = random.choice(list(order_map.keys()))
            batch.append(random.choice(order_map[o]))

            c = random.choice(list(class_map.keys()))
            batch.append(random.choice(class_map[c]))

            p = random.choice(list(phylum_map.keys()))
            batch.append(random.choice(phylum_map[p]))

            while len(batch) < self.batch_size:
                batch.append(random.choice(all_indices))

            order.extend(batch)

        return iter(order)

    def __len__(self):
        return len(samples)

# =========================================================
# tokenizer
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =========================================================
# dataset
# =========================================================

class TaxonomyDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(samples)

    def __getitem__(self, idx):

        s = self.samples[idx]

        enc = tokenizer(
            s["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),

            "taxid": s["taxid"],
            "species": s["species"],

            "phylum": s["phylum"],
            "class": s["class"],
            "order": s["order"],
            "family": s["family"],
            "genus": s["genus"],
        }

def collate_fn(batch):

    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),

        "taxid": [b["taxid"] for b in batch],
        "species": [b["species"] for b in batch],

        "phylum": [b["phylum"] for b in batch],
        "class": [b["class"] for b in batch],
        "order": [b["order"] for b in batch],
        "family": [b["family"] for b in batch],
        "genus": [b["genus"] for b in batch],
    }

dataset = TaxonomyDataset(samples)

sampler = TaxonomyBatchSampler(BATCH_SIZE)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    collate_fn=collate_fn,
    shuffle=False
)

# =========================================================
# model
# =========================================================

class TaxonomyEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

    def forward(self, input_ids, attention_mask):

        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        emb = out.last_hidden_state.mean(dim=1)

        emb = F.normalize(emb, dim=1)

        return emb

model = TaxonomyEncoder().to(DEVICE)

# =========================================================
# metric loss
# =========================================================

def metric_loss(
    emb,
    genus,
    family,
    order,
    class_,
    phylum
):

    B = emb.size(0)

    loss = 0.0
    cnt = 0

    for i in range(B):

        for j in range(i + 1, B):

            d = torch.norm(emb[i] - emb[j])

            if genus[i] == genus[j]:

                loss += F.relu(0.25 - d) + F.relu(d - 0.45)

            elif family[i] == family[j]:

                loss += F.relu(0.45 - d) + F.relu(d - 0.65)

            elif order[i] == order[j]:

                loss += F.relu(0.65 - d) + F.relu(d - 0.85)

            elif class_[i] == class_[j]:

                loss += F.relu(0.85 - d) + F.relu(d - 1.00)

            elif phylum[i] == phylum[j]:

                loss += F.relu(1.00 - d) + F.relu(d - 1.15)

            else:

                loss += F.relu(1.25 - d)

            cnt += 1

    return loss / max(cnt, 1)

# =========================================================
# optimizer
# =========================================================

optimizer = AdamW(model.parameters(), lr=LR)

total_steps = len(dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    int(0.1 * total_steps),
    total_steps
)

# =========================================================
# train
# =========================================================

print("\n=== METRIC LEARNING ===\n")

model.train()

for epoch in range(EPOCHS):

    total_mt = 0.0
    step = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for batch in pbar:

        x = batch["input_ids"].to(DEVICE)
        m = batch["attention_mask"].to(DEVICE)

        emb = model(x, m)

        mt = metric_loss(
            emb,
            batch["genus"],
            batch["family"],
            batch["order"],
            batch["class"],
            batch["phylum"]
        )

        optimizer.zero_grad()

        mt.backward()

        optimizer.step()

        scheduler.step()

        total_mt += mt.item()
        step += 1

        pbar.set_postfix({
            "metric": f"{mt.item():.4f}"
        })

    print(
        f"\nEpoch {epoch+1}"
        f"\nAvg Metric Loss: {total_mt/step:.6f}\n"
    )

# =========================================================
# save
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

save_file(
    model.state_dict(),
    os.path.join(OUTPUT_DIR, "model.safetensors")
)

tokenizer.save_pretrained(OUTPUT_DIR)

# =========================================================
# extract embeddings
# =========================================================

print("\nExtracting embeddings...\n")

model.eval()

all_emb = []
all_taxid = []
all_species = []

with torch.no_grad():

    for batch in tqdm(dataloader):

        x = batch["input_ids"].to(DEVICE)
        m = batch["attention_mask"].to(DEVICE)

        emb = model(x, m)

        all_emb.append(emb.cpu().numpy())

        all_taxid.extend(batch["taxid"])

        all_species.extend(batch["species"])

all_emb = np.concatenate(all_emb, axis=0)

np.save(
    os.path.join(OUTPUT_DIR, "microbe_embeddings.npy"),
    all_emb
)

with open(
    os.path.join(OUTPUT_DIR, "microbe_taxids.txt"),
    "w"
) as f:

    for t in all_taxid:
        f.write(str(t) + "\n")

with open(
    os.path.join(OUTPUT_DIR, "microbe_species.txt"),
    "w",
    encoding="utf-8"
) as f:

    for s in all_species:
        f.write(s + "\n")

print("\nAll outputs saved.\n")
