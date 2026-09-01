"""
Embedding feasibility probe: test if text embeddings predict paper-reported signals.
Probes: A) disease (sanity), B) sequencing method (concrete fact), C) enriched/depleted direction.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")

# Load papers
emily_papers = json.load(open('../EmilySong_GoldStandardPaper/all_usable_papers.json'))
new_papers = json.load(open('new_papers.json'))
extractions = json.load(open('extractions_corrected.json'))
graph_data = json.load(open('graph.json'))

print(f"Loaded: {len(emily_papers)} Emily papers, {len(new_papers)} new papers, "
      f"{len(extractions)} extractions")

# Build a paper index: title -> (full_text, metadata_dict)
# For emily papers, link and title are keys; for new papers, use title/link
paper_index = {}

for paper in emily_papers:
    # Use title as primary key
    key = paper.get('title', '').strip()
    if key:
        paper_index[key] = {
            'text': paper.get('text', ''),
            'disease': paper.get('disease', ''),
            'sequencing': paper.get('sequencing', None),
            'source': 'emily'
        }

for paper in new_papers:
    key = paper.get('title', '').strip()
    if key:
        paper_index[key] = {
            'text': paper.get('text', ''),
            'disease': paper.get('disease', ''),
            'sequencing': paper.get('sequencing', None),
            'source': 'new'
        }

print(f"Paper index: {len(paper_index)} unique papers")

# Map extractions to papers in our index
extraction_to_paper_id = {}  # extraction idx -> paper title
extraction_meta = {}  # extraction idx -> (disease, sequencing)

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

print(f"Matched {matched}/{len(extractions)} extractions to paper index")

# Embed papers
print("\nEmbedding papers with all-MiniLM-L6-v2...")
model = SentenceTransformer('all-MiniLM-L6-v2')

paper_embeddings = {}  # paper_title -> embedding

for title, meta in paper_index.items():
    text = meta['text']
    if not text:
        continue

    # Chunk strategy: take first ~2000 chars (abstract-ish head) and ~2000 chars from middle (methods-ish)
    # Then mean-pool the two chunk embeddings
    chunks = []

    # First chunk: first 2000 chars
    chunk1 = text[:2000]
    chunks.append(chunk1)

    # Second chunk: middle section (around 40-60% through the text)
    mid_start = int(len(text) * 0.4)
    mid_end = min(mid_start + 2000, len(text))
    if mid_end > mid_start:
        chunk2 = text[mid_start:mid_end]
        chunks.append(chunk2)

    # Embed chunks and mean-pool
    embeddings = model.encode(chunks, convert_to_numpy=True)
    mean_embedding = np.mean(embeddings, axis=0)
    paper_embeddings[title] = mean_embedding

print(f"Embedded {len(paper_embeddings)} papers")

# Prepare data for probes
# Build X (embeddings), y (labels), and paper_ids (for stratification)

# --- PROBE A: DISEASE (sanity check) ---
print("\n" + "="*60)
print("PROBE A: Disease prediction (sanity check)")
print("="*60)

X_disease = []
y_disease = []
paper_ids_disease = []

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
    paper_ids_disease.append(paper_title)
    disease_counts[disease] += 1

# Filter to diseases with >= 15 papers
print(f"Disease counts (before filter): {len(disease_counts)} diseases")
print(disease_counts.most_common(10))

valid_diseases = set(d for d, c in disease_counts.items() if c >= 15)
print(f"Diseases with >=15 papers: {len(valid_diseases)}")

# Keep track of valid indices before filtering arrays
valid_indices = [i for i in range(len(y_disease)) if y_disease[i] in valid_diseases]
X_disease = np.array([X_disease[i] for i in valid_indices])
y_disease = np.array([y_disease[i] for i in valid_indices])
paper_ids_disease = np.array([paper_ids_disease[i] for i in valid_indices])

print(f"X_disease shape: {X_disease.shape}, unique diseases: {len(set(y_disease))}")

# Encode labels
le_disease = LabelEncoder()
y_disease_encoded = le_disease.fit_transform(y_disease)

# 5-fold CV with logistic regression
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies_disease = []

for train_idx, test_idx in kf.split(X_disease):
    X_train, X_test = X_disease[train_idx], X_disease[test_idx]
    y_train, y_test = y_disease_encoded[train_idx], y_disease_encoded[test_idx]

    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    accuracies_disease.append(acc)

disease_acc_mean = np.mean(accuracies_disease)
disease_acc_std = np.std(accuracies_disease)
print(f"5-fold CV accuracy: {disease_acc_mean:.4f} ± {disease_acc_std:.4f}")

# Permutation test
n_perms = 100  # Reduced from 200 for speed
perm_accs_disease = []
for perm_idx in range(n_perms):
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

print(f"Permutation test (n={n_perms}):")
print(f"  Observed accuracy: {disease_acc_mean:.4f}")
print(f"  Null mean: {perm_mean_disease:.4f}")
print(f"  Null 95th percentile: {perm_p95_disease:.4f}")
print(f"  P-value: {p_value_disease:.4f}")
print(f"  Result: {'PASS' if disease_acc_mean > perm_p95_disease else 'FAIL'}")

# --- PROBE B: SEQUENCING (concrete fact) ---
print("\n" + "="*60)
print("PROBE B: Sequencing method prediction (concrete fact)")
print("="*60)

X_seq = []
y_seq = []
paper_ids_seq = []

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
    paper_ids_seq.append(paper_title)

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
paper_ids_seq = np.array(paper_ids_seq)

print(f"Sequencing samples: {len(y_seq)}")
print(f"Sequencing types: {Counter(y_seq)}")

# Encode labels (binary classification likely)
le_seq = LabelEncoder()
y_seq_encoded = le_seq.fit_transform(y_seq)

# 5-fold CV
accuracies_seq = []
for train_idx, test_idx in kf.split(X_seq):
    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq_encoded[train_idx], y_seq_encoded[test_idx]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    accuracies_seq.append(acc)

seq_acc_mean = np.mean(accuracies_seq)
seq_acc_std = np.std(accuracies_seq)
print(f"5-fold CV accuracy: {seq_acc_mean:.4f} ± {seq_acc_std:.4f}")

# Permutation test
perm_accs_seq = []
for perm_idx in range(n_perms):
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

print(f"Permutation test (n={n_perms}):")
print(f"  Observed accuracy: {seq_acc_mean:.4f}")
print(f"  Null mean: {perm_mean_seq:.4f}")
print(f"  Null 95th percentile: {perm_p95_seq:.4f}")
print(f"  P-value: {p_value_seq:.4f}")
print(f"  Result: {'PASS' if seq_acc_mean > perm_p95_seq else 'FAIL'}")

# --- PROBE C: ENRICHED/DEPLETED DIRECTION ---
print("\n" + "="*60)
print("PROBE C: Enriched/Depleted direction prediction")
print("="*60)

# Get contested edges from graph.json
contested_edges = [e for e in graph_data['edges'] if e.get('contested', False)]
print(f"Total contested edges: {len(contested_edges)}")

# Build observations: (paper_id, direction) pairs
X_direction = []
y_direction = []
paper_ids_direction = []  # Track which paper each observation came from

for edge in contested_edges:
    ev_list = edge.get('ev', [])
    for ev in ev_list:
        paper_idx = ev.get('i')
        direction = ev.get('d')  # 'e' for enriched, 'd' for depleted

        if paper_idx is None or direction is None:
            continue
        if paper_idx >= len(graph_data['papers']):
            continue

        paper_title = graph_data['papers'][paper_idx]
        if paper_title not in paper_embeddings:
            continue

        X_direction.append(paper_embeddings[paper_title])
        y_direction.append(1 if direction == 'e' else 0)  # 1=enriched, 0=depleted
        paper_ids_direction.append(paper_title)

X_direction = np.array(X_direction)
y_direction = np.array(y_direction)
paper_ids_direction = np.array(paper_ids_direction)

print(f"Total observations: {len(y_direction)}")
print(f"Enriched: {np.sum(y_direction)}, Depleted: {len(y_direction) - np.sum(y_direction)}")
print(f"Unique papers in observations: {len(set(paper_ids_direction))}")

# 5-fold CV STRATIFIED BY PAPER
# Create a mapping from paper to fold
unique_papers_direction = list(set(paper_ids_direction))
paper_to_fold = {}
fold_assignment = np.array_split(unique_papers_direction, 5)
for fold_idx, papers in enumerate(fold_assignment):
    for paper in papers:
        paper_to_fold[paper] = fold_idx

# Manual 5-fold CV with paper-level stratification
accuracies_direction = []
for fold_idx in range(5):
    test_mask = np.array([paper_to_fold[paper_ids_direction[i]] == fold_idx
                          for i in range(len(paper_ids_direction))])
    train_mask = ~test_mask

    X_train, X_test = X_direction[train_mask], X_direction[test_mask]
    y_train, y_test = y_direction[train_mask], y_direction[test_mask]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        # Skip if one class is missing
        continue

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    accuracies_direction.append(acc)

dir_acc_mean = np.mean(accuracies_direction)
dir_acc_std = np.std(accuracies_direction)
print(f"5-fold CV accuracy (paper-stratified): {dir_acc_mean:.4f} ± {dir_acc_std:.4f}")

# Permutation test (also paper-stratified)
perm_accs_direction = []
for perm_idx in range(n_perms):
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

print(f"Permutation test (n={n_perms}):")
print(f"  Observed accuracy: {dir_acc_mean:.4f}")
print(f"  Null mean: {perm_mean_direction:.4f}")
print(f"  Null 95th percentile: {perm_p95_direction:.4f}")
print(f"  P-value: {p_value_direction:.4f}")
print(f"  Result: {'PASS' if dir_acc_mean > perm_p95_direction else 'FAIL'}")

# Save results
results = {
    'probe_A_disease': {
        'n_samples': len(X_disease),
        'accuracy': disease_acc_mean,
        'null_mean': perm_mean_disease,
        'null_p95': perm_p95_disease,
        'p_value': p_value_disease,
        'verdict': 'PASS' if disease_acc_mean > perm_p95_disease else 'FAIL'
    },
    'probe_B_sequencing': {
        'n_samples': len(X_seq),
        'accuracy': seq_acc_mean,
        'null_mean': perm_mean_seq,
        'null_p95': perm_p95_seq,
        'p_value': p_value_seq,
        'verdict': 'PASS' if seq_acc_mean > perm_p95_seq else 'FAIL'
    },
    'probe_C_direction': {
        'n_samples': len(X_direction),
        'accuracy': dir_acc_mean,
        'null_mean': perm_mean_direction,
        'null_p95': perm_p95_direction,
        'p_value': p_value_direction,
        'verdict': 'PASS' if dir_acc_mean > perm_p95_direction else 'FAIL'
    }
}

with open('probe_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("Results saved to probe_results.json")
print("="*60)
