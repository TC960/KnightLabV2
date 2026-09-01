"""
Fast embedding feasibility probe: test if text embeddings predict paper-reported signals.
Probes: A) disease (sanity), B) sequencing method (concrete fact), C) enriched/depleted direction.
Uses reduced permutations and streamlined CV for speed.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

print("Loading data...", flush=True)

# Load papers
emily_papers = json.load(open('../EmilySong_GoldStandardPaper/all_usable_papers.json'))
new_papers = json.load(open('new_papers.json'))
extractions = json.load(open('extractions_corrected.json'))
graph_data = json.load(open('graph.json'))

print(f"Loaded: {len(emily_papers)} Emily papers, {len(new_papers)} new papers, {len(extractions)} extractions", flush=True)

# Build paper index
paper_index = {}

for paper in emily_papers:
    key = paper.get('title', '').strip()
    if key:
        paper_index[key] = {
            'text': paper.get('text', ''),
            'disease': paper.get('disease', ''),
            'sequencing': paper.get('sequencing', None),
        }

for paper in new_papers:
    key = paper.get('title', '').strip()
    if key:
        paper_index[key] = {
            'text': paper.get('text', ''),
            'disease': paper.get('disease', ''),
            'sequencing': paper.get('sequencing', None),
        }

print(f"Paper index: {len(paper_index)} unique papers", flush=True)

# Map extractions to papers
extraction_to_paper_id = {}
extraction_meta = {}

matched = 0
for i, ext in enumerate(extractions):
    title = ext.get('title', '').strip()
    if title in paper_index:
        extraction_to_paper_id[i] = title
        extraction_meta[i] = {
            'disease': ext.get('disease'),
            'sequencing': ext.get('sequencing')
        }
        matched += 1

print(f"Matched {matched}/{len(extractions)} extractions to paper index", flush=True)

# Embed papers
print("\nEmbedding papers with all-MiniLM-L6-v2...", flush=True)
model = SentenceTransformer('all-MiniLM-L6-v2')

paper_embeddings = {}

for idx, (title, meta) in enumerate(paper_index.items()):
    if (idx + 1) % 50 == 0:
        print(f"  Embedded {idx + 1}/{len(paper_index)} papers...", flush=True)

    text = meta['text']
    if not text:
        continue

    # Chunk: first ~2000 chars (abstract-ish) + middle ~2000 chars (methods-ish)
    chunks = []
    chunk1 = text[:2000]
    chunks.append(chunk1)

    mid_start = int(len(text) * 0.4)
    mid_end = min(mid_start + 2000, len(text))
    if mid_end > mid_start:
        chunk2 = text[mid_start:mid_end]
        chunks.append(chunk2)

    embeddings = model.encode(chunks, convert_to_numpy=True)
    mean_embedding = np.mean(embeddings, axis=0)
    paper_embeddings[title] = mean_embedding

print(f"Embedded {len(paper_embeddings)} papers", flush=True)

# --- PROBE A: DISEASE (sanity check) ---
print("\n" + "="*60, flush=True)
print("PROBE A: Disease prediction (sanity check)", flush=True)
print("="*60, flush=True)

X_disease = []
y_disease = []

disease_counts = Counter()
for i, ext in enumerate(extractions):
    if i not in extraction_to_paper_id:
        continue
    paper_title = extraction_to_paper_id[i]
    if paper_title not in paper_embeddings:
        continue

    disease = extraction_meta[i]['disease']
    if not disease:
        continue

    X_disease.append(paper_embeddings[paper_title])
    y_disease.append(disease)
    disease_counts[disease] += 1

print(f"Disease counts: {len(disease_counts)} diseases", flush=True)
print("Top 10:", disease_counts.most_common(10), flush=True)

valid_diseases = set(d for d, c in disease_counts.items() if c >= 15)
print(f"Diseases with >=15 papers: {len(valid_diseases)}", flush=True)

# Filter
valid_indices = [i for i in range(len(y_disease)) if y_disease[i] in valid_diseases]
X_disease = np.array([X_disease[i] for i in valid_indices])
y_disease = np.array([y_disease[i] for i in valid_indices])

print(f"Filtered to {len(X_disease)} samples", flush=True)

le_disease = LabelEncoder()
y_disease_encoded = le_disease.fit_transform(y_disease)

# 5-fold CV
print("Running 5-fold CV...", flush=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies_disease = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_disease)):
    X_train, X_test = X_disease[train_idx], X_disease[test_idx]
    y_train, y_test = y_disease_encoded[train_idx], y_disease_encoded[test_idx]

    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    accuracies_disease.append(acc)
    print(f"  Fold {fold+1}: {acc:.4f}", flush=True)

disease_acc_mean = np.mean(accuracies_disease)
disease_acc_std = np.std(accuracies_disease)
print(f"5-fold CV accuracy: {disease_acc_mean:.4f} ± {disease_acc_std:.4f}", flush=True)

# Permutation test (50 perms)
print("Permutation test (n=50)...", flush=True)
n_perms = 50
perm_accs_disease = []

for perm_idx in range(n_perms):
    if (perm_idx + 1) % 10 == 0:
        print(f"  Perm {perm_idx + 1}/{n_perms}...", flush=True)

    y_shuffled = y_disease_encoded.copy()
    np.random.shuffle(y_shuffled)

    fold_accs = []
    for train_idx, test_idx in kf.split(X_disease):
        X_train, X_test = X_disease[train_idx], X_disease[test_idx]
        y_train, y_test = y_shuffled[train_idx], y_shuffled[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        fold_accs.append(acc)

    perm_accs_disease.append(np.mean(fold_accs))

perm_mean_disease = np.mean(perm_accs_disease)
perm_p95_disease = np.percentile(perm_accs_disease, 95)
p_value_disease = np.mean(np.array(perm_accs_disease) >= disease_acc_mean)

print(f"Observed accuracy: {disease_acc_mean:.4f}")
print(f"Null mean: {perm_mean_disease:.4f}, 95th percentile: {perm_p95_disease:.4f}")
print(f"P-value: {p_value_disease:.4f}")
print(f"Result: {'PASS' if disease_acc_mean > perm_p95_disease else 'FAIL'}", flush=True)

# --- PROBE B: SEQUENCING ---
print("\n" + "="*60, flush=True)
print("PROBE B: Sequencing method prediction (concrete fact)", flush=True)
print("="*60, flush=True)

X_seq = []
y_seq = []

for i, ext in enumerate(extractions):
    if i not in extraction_to_paper_id:
        continue
    paper_title = extraction_to_paper_id[i]
    if paper_title not in paper_embeddings:
        continue

    sequencing = extraction_meta[i]['sequencing']
    if not sequencing:
        continue

    X_seq.append(paper_embeddings[paper_title])
    y_seq.append(sequencing)

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

print(f"Sequencing samples: {len(y_seq)}", flush=True)
print(f"Sequencing types: {Counter(y_seq)}", flush=True)

le_seq = LabelEncoder()
y_seq_encoded = le_seq.fit_transform(y_seq)

print("Running 5-fold CV...", flush=True)
accuracies_seq = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_seq)):
    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq_encoded[train_idx], y_seq_encoded[test_idx]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    accuracies_seq.append(acc)
    print(f"  Fold {fold+1}: {acc:.4f}", flush=True)

seq_acc_mean = np.mean(accuracies_seq)
seq_acc_std = np.std(accuracies_seq)
print(f"5-fold CV accuracy: {seq_acc_mean:.4f} ± {seq_acc_std:.4f}", flush=True)

# Permutation test
print("Permutation test (n=50)...", flush=True)
perm_accs_seq = []
for perm_idx in range(n_perms):
    if (perm_idx + 1) % 10 == 0:
        print(f"  Perm {perm_idx + 1}/{n_perms}...", flush=True)

    y_shuffled = y_seq_encoded.copy()
    np.random.shuffle(y_shuffled)

    fold_accs = []
    for train_idx, test_idx in kf.split(X_seq):
        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y_shuffled[train_idx], y_shuffled[test_idx]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        fold_accs.append(acc)

    perm_accs_seq.append(np.mean(fold_accs))

perm_mean_seq = np.mean(perm_accs_seq)
perm_p95_seq = np.percentile(perm_accs_seq, 95)
p_value_seq = np.mean(np.array(perm_accs_seq) >= seq_acc_mean)

print(f"Observed accuracy: {seq_acc_mean:.4f}")
print(f"Null mean: {perm_mean_seq:.4f}, 95th percentile: {perm_p95_seq:.4f}")
print(f"P-value: {p_value_seq:.4f}")
print(f"Result: {'PASS' if seq_acc_mean > perm_p95_seq else 'FAIL'}", flush=True)

# --- PROBE C: ENRICHED/DEPLETED DIRECTION ---
print("\n" + "="*60, flush=True)
print("PROBE C: Enriched/Depleted direction prediction", flush=True)
print("="*60, flush=True)

contested_edges = [e for e in graph_data['edges'] if e.get('contested', False)]
print(f"Total contested edges: {len(contested_edges)}", flush=True)

X_direction = []
y_direction = []
paper_ids_direction = []

for edge in contested_edges:
    ev_list = edge.get('ev', [])
    for ev in ev_list:
        paper_idx = ev.get('i')
        direction = ev.get('d')  # 'e' or 'd'

        if paper_idx is None or direction is None or paper_idx >= len(graph_data['papers']):
            continue

        # papers is a list of dicts with 'title' field
        paper_record = graph_data['papers'][paper_idx]
        if isinstance(paper_record, dict):
            paper_title = paper_record.get('title')
        else:
            paper_title = paper_record

        if not paper_title or paper_title not in paper_embeddings:
            continue

        X_direction.append(paper_embeddings[paper_title])
        y_direction.append(1 if direction == 'e' else 0)
        paper_ids_direction.append(paper_title)

X_direction = np.array(X_direction)
y_direction = np.array(y_direction)
paper_ids_direction = np.array(paper_ids_direction)

print(f"Total observations: {len(y_direction)}", flush=True)
print(f"Enriched: {np.sum(y_direction)}, Depleted: {len(y_direction) - np.sum(y_direction)}", flush=True)
print(f"Unique papers: {len(set(paper_ids_direction))}", flush=True)

# 5-fold CV stratified by PAPER
print("Running 5-fold CV (stratified by paper)...", flush=True)
unique_papers_direction = list(set(paper_ids_direction))
paper_to_fold = {}
fold_assignment = np.array_split(unique_papers_direction, 5)
for fold_idx, papers in enumerate(fold_assignment):
    for paper in papers:
        paper_to_fold[paper] = fold_idx

accuracies_direction = []
for fold_idx in range(5):
    test_mask = np.array([paper_to_fold[paper_ids_direction[i]] == fold_idx
                          for i in range(len(paper_ids_direction))])
    train_mask = ~test_mask

    X_train, X_test = X_direction[train_mask], X_direction[test_mask]
    y_train, y_test = y_direction[train_mask], y_direction[test_mask]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        continue

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    accuracies_direction.append(acc)
    print(f"  Fold {fold_idx+1}: {acc:.4f}", flush=True)

dir_acc_mean = np.mean(accuracies_direction)
dir_acc_std = np.std(accuracies_direction)
print(f"5-fold CV accuracy: {dir_acc_mean:.4f} ± {dir_acc_std:.4f}", flush=True)

# Permutation test
print("Permutation test (n=50)...", flush=True)
perm_accs_direction = []
for perm_idx in range(n_perms):
    if (perm_idx + 1) % 10 == 0:
        print(f"  Perm {perm_idx + 1}/{n_perms}...", flush=True)

    y_shuffled = y_direction.copy()
    np.random.shuffle(y_shuffled)

    fold_accs = []
    for fold_idx in range(5):
        test_mask = np.array([paper_to_fold[paper_ids_direction[i]] == fold_idx
                              for i in range(len(paper_ids_direction))])
        train_mask = ~test_mask

        X_train, X_test = X_direction[train_mask], X_direction[test_mask]
        y_train, y_test = y_shuffled[train_mask], y_shuffled[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        fold_accs.append(acc)

    if fold_accs:
        perm_accs_direction.append(np.mean(fold_accs))

perm_mean_direction = np.mean(perm_accs_direction)
perm_p95_direction = np.percentile(perm_accs_direction, 95)
p_value_direction = np.mean(np.array(perm_accs_direction) >= dir_acc_mean)

print(f"Observed accuracy: {dir_acc_mean:.4f}")
print(f"Null mean: {perm_mean_direction:.4f}, 95th percentile: {perm_p95_direction:.4f}")
print(f"P-value: {p_value_direction:.4f}")
print(f"Result: {'PASS' if dir_acc_mean > perm_p95_direction else 'FAIL'}", flush=True)

# Save results
print("\nSaving results...", flush=True)
results = {
    'probe_A_disease': {
        'n_samples': len(X_disease),
        'accuracy': float(disease_acc_mean),
        'null_mean': float(perm_mean_disease),
        'null_p95': float(perm_p95_disease),
        'p_value': float(p_value_disease),
        'verdict': 'PASS' if disease_acc_mean > perm_p95_disease else 'FAIL'
    },
    'probe_B_sequencing': {
        'n_samples': len(X_seq),
        'accuracy': float(seq_acc_mean),
        'null_mean': float(perm_mean_seq),
        'null_p95': float(perm_p95_seq),
        'p_value': float(p_value_seq),
        'verdict': 'PASS' if seq_acc_mean > perm_p95_seq else 'FAIL'
    },
    'probe_C_direction': {
        'n_samples': len(X_direction),
        'accuracy': float(dir_acc_mean),
        'null_mean': float(perm_mean_direction),
        'null_p95': float(perm_p95_direction),
        'p_value': float(p_value_direction),
        'verdict': 'PASS' if dir_acc_mean > perm_p95_direction else 'FAIL'
    }
}

with open('probe_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to probe_results.json", flush=True)
print("\n" + "="*60, flush=True)
print("SUMMARY", flush=True)
print("="*60, flush=True)

for probe, data in results.items():
    print(f"\n{probe}:")
    print(f"  n={data['n_samples']}")
    print(f"  accuracy={data['accuracy']:.4f}")
    print(f"  null mean/p95: {data['null_mean']:.4f} / {data['null_p95']:.4f}")
    print(f"  p-value: {data['p_value']:.4f}")
    print(f"  verdict: {data['verdict']}")
