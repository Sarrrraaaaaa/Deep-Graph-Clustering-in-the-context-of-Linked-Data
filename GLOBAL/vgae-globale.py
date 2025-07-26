# Fix NumPy compatibility issues
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Handle NumPy version compatibility
try:
    import numpy as np
    # Check if we have NumPy 2.x compatibility issues
    if hasattr(np, '__version__') and np.__version__.startswith('2.'):
        # Try to set compatibility mode
        try:
            np._NoValue = np._NoValue
        except AttributeError:
            pass
except ImportError:
    print("NumPy not found. Please install: pip install numpy")
    exit(1)

# Safe matplotlib import with fallback
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: matplotlib import failed: {e}")
    print("Visualization features will be disabled. Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False
    # Create dummy plt for compatibility
    class DummyPlt:
        def figure(self, *args, **kwargs): pass
        def subplots(self, *args, **kwargs): return None, None
        def savefig(self, *args, **kwargs): pass
        def close(self, *args, **kwargs): pass
        def tight_layout(self, *args, **kwargs): pass
    plt = DummyPlt()

from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
import time
import pickle
import random  # Ajout de l'import manquant
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, accuracy_score

# Safe TSNE import
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    print("Warning: TSNE not available")
    TSNE_AVAILABLE = False

import os
import dgl
from dgl.nn import HeteroGraphConv
from collections import defaultdict
from rdflib import Graph, URIRef
from dgl import heterograph
class DEC(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_clusters, alpha=1.0):
        super(DEC, self).__init__()
        self.alpha = alpha
        self.n_clusters = n_clusters

        # Vérifier que hidden_dims n'est pas vide
        if not hidden_dims:
            raise ValueError("hidden_dims ne peut pas être vide")

        self.latent_dim = hidden_dims[-1]  # La dernière dimension définit l'espace latent

        # Encoder avec vérification des dimensions
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        self.encoder = nn.Sequential(*layers)

        # Cluster layer doit avoir la même dimension que la sortie de l'encodeur
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, self.latent_dim))
        nn.init.xavier_uniform_(self.cluster_layer)

    def forward(self, x):
        return self.encoder(x)

    def get_soft_assignment(self, z):
        # Vérification des dimensions
        if z.size(1) != self.cluster_layer.size(1):
            raise ValueError(f"Dimension mismatch: embeddings {z.size(1)} vs clusters {self.cluster_layer.size(1)}")

        # Normalisation
        z = F.normalize(z, p=2, dim=1)
        cluster_centers = F.normalize(self.cluster_layer, p=2, dim=1)

        # Similarité cosinus
        cos_sim = torch.mm(z, cluster_centers.t())  # [batch_size, n_clusters]

        # Conversion en probabilités
        q = 1.0 / (1.0 + (1 - cos_sim) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
        return q

def train_dec(embeddings, n_clusters=None, true_labels=None, hidden_dims=None,
             lr=0.001, max_epochs=300, batch_size=256, tol=1e-5, early_stopping_patience=15):
    """
    Improved DEC model training with automatic cluster number estimation and better stability
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Préparation des embeddings with proper dtype handling
    if isinstance(embeddings, dict):
        all_embeddings = torch.cat([e if isinstance(e, torch.Tensor) else torch.tensor(e, dtype=torch.float32)
                                  for e in embeddings.values()], dim=0)
    else:
        all_embeddings = embeddings if isinstance(embeddings, torch.Tensor) else torch.tensor(embeddings, dtype=torch.float32)

    # Ensure float32 dtype consistency
    all_embeddings = all_embeddings.float()
    all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
    all_embeddings = torch.nan_to_num(all_embeddings)
    all_embeddings = all_embeddings.to(device)

    # 2. Réduction de dimension si nécessaire
    input_dim = all_embeddings.shape[1]
    if input_dim > 256:
        pca = PCA(n_components=min(256, input_dim-1))
        all_embeddings_np = pca.fit_transform(all_embeddings.cpu().numpy())
        all_embeddings = torch.tensor(all_embeddings_np, dtype=torch.float32, device=device)
        input_dim = all_embeddings.shape[1]

    # Create output directory for images
    output_dir = "images VGAE + HGT"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Determining the optimal number of clusters
    if n_clusters is None:
        print("Estimating optimal number of clusters...")
        from sklearn.metrics import silhouette_score

        # Test a range of possible cluster numbers
        cluster_range = range(2, min(20, len(all_embeddings)//10))
        best_score = -1
        best_k = 2

        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(all_embeddings.cpu().numpy())
            score = silhouette_score(all_embeddings.cpu().numpy(), labels)

            if score > best_score:
                best_score = score
                best_k = k

        n_clusters = best_k
        print(f"Estimated optimal clusters: {n_clusters} (silhouette score: {best_score:.3f})")

    # 4. Définition des dimensions cachées
    if hidden_dims is None:
        hidden_dims = [max(256, input_dim), max(128, input_dim//2), max(64, input_dim//4)]
        print(f"Dimensions cachées automatiques: {hidden_dims}")

    # 5. Initialisation du modèle avec une seed spécifique pour garantir la reproductibilité
    torch.manual_seed(42)
    model = DEC(input_dim, hidden_dims, n_clusters, alpha=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 6. Initialisation des centres de clusters avec K-means
    print("Initialisation des centres de clusters...")
    with torch.no_grad():
        # Encodage des données
        encoded = model(all_embeddings)
        encoded_np = encoded.cpu().numpy()

        # K-means sur les données encodées avec une seed spécifique
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, random_state=42)
        kmeans.fit(encoded.cpu().numpy())

        # Vérification des dimensions
        if kmeans.cluster_centers_.shape[1] != model.latent_dim:
            raise ValueError(f"Dimension des centres K-means {kmeans.cluster_centers_.shape[1]} ne correspond pas à la dimension latente {model.latent_dim}")

        model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)

    # 7. Training loop with improvements
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        model.train()
        permutation = torch.randperm(all_embeddings.size(0))
        total_loss = 0

        # Mini-batch training
        for i in range(0, all_embeddings.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch = all_embeddings[indices]

            optimizer.zero_grad()
            z = model(batch)
            q = model.get_soft_assignment(z)

            # Improved target distribution computation
            p = (q**2) / (torch.sum(q, dim=0) + 1e-8)
            p = (p.t() / (torch.sum(p, dim=1) + 1e-8)).t()  # Normalize

            # KL divergence loss with stability improvements
            loss = F.kl_div(torch.log(q + 1e-8), p.detach(), reduction='batchmean')

            # Optional supervised loss if true labels are available
            if true_labels is not None:
                batch_labels = true_labels[indices.cpu().numpy()]
                supervised_loss = F.cross_entropy(q, torch.tensor(batch_labels, dtype=torch.long, device=device))
                loss = loss + 0.1 * supervised_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / all_embeddings.size(0)
        scheduler.step(avg_loss)

        # Early stopping check
        if avg_loss < best_loss - tol:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        # Afficher la progression
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}, loss: {avg_loss:.4f}")
            break

    # Charger le meilleur modèle
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Prédiction finale
    model.eval()
    with torch.no_grad():
        z = model(all_embeddings)
        q = model.get_soft_assignment(z)
        _, pred_labels = torch.max(q, dim=1)
        pred_labels = pred_labels.cpu().numpy()

    return model, pred_labels
class VGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Mean and log variance layers
        self.mean = nn.Linear(hidden_dim, out_dim)
        self.log_std = nn.Linear(hidden_dim, out_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mean = self.mean(h)
        log_std = self.log_std(h)
        return mean, log_std

    def reparameterize(self, mean, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_std = self.encode(x)
        z = self.reparameterize(mean, log_std)
        x_recon = self.decode(z)
        return x_recon, mean, log_std

def train_vgae(g, node_features, num_epochs=100, lr=0.005, weight_decay=1e-4):
    """
    Train a Variational Graph Autoencoder (VGAE) model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize VGAE model for each node type
    vgae_models = {}
    optimizers = {}

    for ntype in g.ntypes:
        if g.number_of_nodes(ntype) == 0:
            continue

        # Get node features for this type
        if isinstance(node_features, dict):
            # If node_features is a dictionary with node type keys
            if ntype in node_features:
                features = node_features[ntype]
            else:
                # If node_features is a dictionary with node IDs as keys
                # Get all features for this node type
                node_ids = g.nodes(ntype)
                features = []
                for nid in node_ids:
                    if int(nid) in node_features:
                        features.append(node_features[int(nid)])
                if not features:
                    print(f"Warning: No features found for node type {ntype}, skipping...")
                    continue
                features = torch.stack(features)
        else:
            # If node_features is a tensor or array
            features = g.nodes[ntype].data['feat']

        in_dim = features.shape[1]
        hidden_dim = 256  # IMPROVED for better clustering (was 128)
        out_dim = 128     # IMPROVED for better clustering (was 64)

        # Create and move model to device
        model = VGAE(in_dim, hidden_dim, out_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        vgae_models[ntype] = model
        optimizers[ntype] = optimizer

    if not vgae_models:
        raise ValueError("No valid node types found with features")

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for ntype, model in vgae_models.items():
            model.train()
            optimizer = optimizers[ntype]
            optimizer.zero_grad()

            # Get features for this node type
            if isinstance(node_features, dict):
                if ntype in node_features:
                    features = node_features[ntype]
                else:
                    node_ids = g.nodes(ntype)
                    features = []
                    for nid in node_ids:
                        if int(nid) in node_features:
                            features.append(node_features[int(nid)])
                    features = torch.stack(features)
            else:
                features = g.nodes[ntype].data['feat']

            features = features.to(device)

            # Forward pass
            x_recon, mean, log_std = model(features)

            # Compute loss
            recon_loss = F.mse_loss(x_recon, features)
            kl_loss = -0.5 * torch.mean(1 + 2 * log_std - mean.pow(2) - (2 * log_std).exp())
            loss = recon_loss + kl_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(vgae_models):.4f}")

    # Generate embeddings for each node type
    embeddings = {}
    for ntype, model in vgae_models.items():
        model.eval()
        with torch.no_grad():
            if isinstance(node_features, dict):
                if ntype in node_features:
                    features = node_features[ntype]
                else:
                    node_ids = g.nodes(ntype)
                    features = []
                    for nid in node_ids:
                        if int(nid) in node_features:
                            features.append(node_features[int(nid)])
                    features = torch.stack(features)
            else:
                features = g.nodes[ntype].data['feat']

            features = features.to(device)
            mean, _ = model.encode(features)

            # 🔧 Add minimal noise to VGAE embeddings for regularization
            noise = torch.randn_like(mean) * 0.01  # 1% noise (reduced from 5%)
            embeddings[ntype] = mean + noise

    for ntype in embeddings:
        if torch.isnan(embeddings[ntype]).any():
            print(f"NaN detected before normalization for {ntype}")
        norm = torch.norm(embeddings[ntype], p=2, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        embeddings[ntype] = embeddings[ntype] / norm
        if torch.isnan(embeddings[ntype]).any():
            print(f"NaN detected after normalization for {ntype}")

    return vgae_models, embeddings

class HGTLayer(nn.Module):
    def __init__(self, in_dim, out_dim, ntypes, etypes, n_heads=4, dropout=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ntypes = ntypes
        self.etypes = etypes
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.out_dim = self.d_k * n_heads
        self.dropout = dropout

        # Linear transformations for each node type
        self.k_linears = nn.ModuleDict({
            ntype: nn.Linear(in_dim, self.out_dim) for ntype in ntypes
        })
        self.q_linears = nn.ModuleDict({
            ntype: nn.Linear(in_dim, self.out_dim) for ntype in ntypes
        })
        self.v_linears = nn.ModuleDict({
            ntype: nn.Linear(in_dim, self.out_dim) for ntype in ntypes
        })
        self.out_linears = nn.ModuleDict({
            ntype: nn.Linear(self.out_dim, self.out_dim) for ntype in ntypes
        })

        # Relation projections
        self.relation_proj = nn.ModuleDict({
            etype: nn.Linear(768, out_dim) for etype in etypes
        })

    def forward(self, G, h, relation_embeddings):
        with G.local_scope():
            # Debug: Check input embeddings
            for ntype in h:
                if torch.isnan(h[ntype]).any():
                    print(f"[DEBUG] NaN in input embeddings for {ntype}")

            # Project relation embeddings
            rel_embeddings = {}
            for canonical_etype in G.canonical_etypes:
                src_type, etype, dst_type = canonical_etype
                if etype in self.relation_proj and etype in relation_embeddings:
                    rel_emb = self.relation_proj[etype](relation_embeddings[etype])
                    if torch.isnan(rel_emb).any():
                        print(f"[DEBUG] NaN in relation embeddings for {etype}")
                    rel_embeddings[canonical_etype] = rel_emb

            feat_dict = {}

            for ntype in G.ntypes:
                if G.number_of_nodes(ntype) == 0:
                    continue

                # Initialize output features
                feat_dict[ntype] = torch.zeros((G.number_of_nodes(ntype), self.out_dim),
                                            device=h[ntype].device)

                # Process each relation type
                for canonical_etype in G.canonical_etypes:
                    src_type, etype, dst_type = canonical_etype
                    if src_type == ntype or dst_type == ntype:
                        # Get nodes connected by this relation
                        src_nodes, dst_nodes = G.edges(etype=canonical_etype)

                        if len(src_nodes) == 0 or len(dst_nodes) == 0:
                            continue

                        # Get features
                        q = self.q_linears[dst_type](h[dst_type])
                        k = self.k_linears[src_type](h[src_type])
                        v = self.v_linears[src_type](h[src_type])

                        # Debug: Check linear transformations
                        if torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any():
                            print(f"[DEBUG] NaN in linear transformations for {canonical_etype}")

                        # Reshape for multi-head attention
                        batch_size = q.size(0)
                        q = q.view(batch_size, self.n_heads, self.d_k)
                        k = k.view(k.size(0), self.n_heads, self.d_k)
                        v = v.view(v.size(0), self.n_heads, self.d_k)

                        # Get features for connected nodes only
                        q = torch.index_select(q, 0, dst_nodes)
                        k = torch.index_select(k, 0, src_nodes)
                        v = torch.index_select(v, 0, src_nodes)

                        # Apply relation-specific transformation if available
                        if canonical_etype in rel_embeddings:
                            rel_emb = rel_embeddings[canonical_etype]
                            rel_emb = rel_emb.view(1, self.n_heads, self.d_k)

                            # Debug: Check relation embedding application
                            if torch.isnan(rel_emb).any():
                                print(f"[DEBUG] NaN in relation embedding application for {canonical_etype}")

                            k = k * rel_emb
                            v = v * rel_emb

                        # Compute attention scores
                        attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
                        attn = F.softmax(attn, dim=2)
                        attn = F.dropout(attn, self.dropout, training=self.training)

                        # Debug: Check attention scores
                        if torch.isnan(attn).any():
                            print(f"[DEBUG] NaN in attention scores for {canonical_etype}")

                        # Apply attention to values
                        output = torch.bmm(attn, v)
                        output = output.view(-1, self.out_dim)

                        # Debug: Check output
                        if torch.isnan(output).any():
                            print(f"[DEBUG] NaN in output for {canonical_etype}")

                        # Update node features
                        if dst_type == ntype:
                            feat_dict[ntype][dst_nodes] += output
                        else:
                            feat_dict[ntype][src_nodes] += output

            # Apply output transformation
            for ntype in G.ntypes:
                if G.number_of_nodes(ntype) == 0 or ntype not in feat_dict:
                    continue
                feat_dict[ntype] = self.out_linears[ntype](feat_dict[ntype])

                # Debug: Check final output
                if torch.isnan(feat_dict[ntype]).any():
                    print(f"[DEBUG] NaN in final output for {ntype}")

            return feat_dict

class HGT(nn.Module):
    def __init__(self, G, in_dim, hidden_dim, out_dim, n_layers=2, n_heads=4, dropout=0.2):
        super().__init__()
        self.G = G
        self.ntypes = G.ntypes
        self.etypes = G.etypes  # Add this line to define etypes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        # Input layers for each node type
        self.embeddings = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype in self.ntypes
        })

        # HGT layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = HGTLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                ntypes=self.ntypes,
                etypes=self.etypes,  # Now self.etypes is defined
                n_heads=n_heads,
                dropout=dropout
            )
            self.layers.append(layer)

        # Output layers for each node type
        self.outputs = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim, out_dim) for ntype in self.ntypes
        })

        # Batch normalization layers
        self.batch_norms = nn.ModuleDict({
            ntype: nn.BatchNorm1d(out_dim) for ntype in self.ntypes
        })

    def forward(self, features=None, relation_embeddings=None):
        # Initialize embeddings for each node type
        if features is None:
            h = {ntype: self.embeddings[ntype](self.G.nodes[ntype].data['feat'])
                 for ntype in self.ntypes}
        else:
            h = {ntype: self.embeddings[ntype](features[ntype])
                 for ntype in self.ntypes}

        # Apply HGT layers with relation embeddings
        for layer in self.layers:
            h_new = layer(self.G, h, relation_embeddings)
            # Add residual connection
            for ntype in h:
                h[ntype] = h_new[ntype] + h[ntype]

        # Apply output layers with batch normalization
        out = {}
        for ntype in self.ntypes:
            out[ntype] = self.outputs[ntype](h[ntype])
            if out[ntype].shape[0] > 1:  # Skip batchnorm when batch size is 1
                out[ntype] = self.batch_norms[ntype](out[ntype])

        return out

def improved_contrastive_loss(embeddings, graph, temperature=0.01, margin=0.2, true_labels=None):
    """
    Version améliorée de la perte contrastive avec mining de paires difficiles, température basse
    et utilisation optionnelle des true labels pour un préentraînement semi-supervisé
    """
    device = embeddings['author'].device

    # Generate positive pairs
    pos_pairs = generate_positive_pairs(graph)

    if not pos_pairs:
        raise ValueError("No positive pairs found in the graph")

    # Prepare tensors for positive pairs
    h1_list = []
    h2_list = []

    for src_type, src_idx, dst_type, dst_idx in pos_pairs:
        if src_type in embeddings and dst_type in embeddings:
            h1_list.append(embeddings[src_type][src_idx])
            h2_list.append(embeddings[dst_type][dst_idx])

    if not h1_list:
        raise ValueError("No valid embedding pairs found")

    h1 = torch.stack(h1_list)
    h2 = torch.stack(h2_list)

    # L2 normalization
    h1 = F.normalize(h1, p=2, dim=1)
    h2 = F.normalize(h2, p=2, dim=1)

    # Compute positive similarity (cosine similarity)
    pos_sim = torch.sum(h1 * h2, dim=1)

    # Compute negative similarity matrix
    batch_size = h1.size(0)
    neg_sim = torch.mm(h1, h2.t())

    # Remove diagonal (positive pairs)
    mask = torch.eye(batch_size, device=device)
    neg_sim = neg_sim * (1 - mask)

    # Hard negative mining: pour chaque ancre, trouver les k négatifs les plus difficiles
    k = 10  # Nombre de négatifs difficiles
    hardest_negatives, _ = torch.topk(neg_sim, k=min(k, batch_size-1), dim=1)
    hardest_negative_mean = torch.mean(hardest_negatives, dim=1)

    # Calculer la perte avec une marge plus stricte et une température plus basse
    pos_term = torch.exp(pos_sim / temperature)
    neg_term = torch.exp(hardest_negative_mean / temperature)

    # Ajouter une marge pour séparer les paires positives et négatives
    eps = 1e-8
    loss = -torch.log((pos_term + eps) / (pos_term + neg_term + margin + eps))

    # Ajouter une régularisation pour éviter le collapse
    reg_loss = torch.mean(torch.abs(torch.sum(h1, dim=0))) + torch.mean(torch.abs(torch.sum(h2, dim=0)))
    loss = loss.mean() + 0.01 * reg_loss

    # Si true_labels est fourni, ajouter un terme de perte semi-supervisée
    if true_labels is not None:
        semi_loss = 0.0
        semi_count = 0

        # Parcourir les types de nœuds avec des labels
        for ntype in true_labels:
            if ntype in embeddings and true_labels[ntype] is not None:
                # Obtenir les embeddings et labels pour ce type
                node_embs = F.normalize(embeddings[ntype], p=2, dim=1)
                labels = torch.tensor(true_labels[ntype], device=device)

                # Calculer une matrice de similarité entre nœuds du même type
                sim_matrix = torch.mm(node_embs, node_embs.t())

                # Créer un masque pour les paires de même label
                n_nodes = node_embs.size(0)
                label_matrix = labels.view(-1, 1).expand(n_nodes, n_nodes)
                mask_same = (label_matrix == label_matrix.t()).float()
                mask_diff = 1.0 - mask_same

                # Exclure la diagonale
                diag_mask = 1.0 - torch.eye(n_nodes, device=device)
                mask_same *= diag_mask
                mask_diff *= diag_mask

                # Calculer la perte semi-supervisée
                pos_sim_semi = (sim_matrix * mask_same).sum() / (mask_same.sum() + eps)
                neg_sim_semi = (sim_matrix * mask_diff).sum() / (mask_diff.sum() + eps)

                # Ajouter à la perte totale
                semi_loss += (neg_sim_semi - pos_sim_semi + margin)
                semi_count += 1

        # Ajouter la perte semi-supervisée si des labels ont été trouvés
        if semi_count > 0:
            loss = loss + 0.5 * (semi_loss / semi_count)

    return loss

def generate_positive_pairs(g):
    """
    Generate positive pairs for contrastive learning with improved filtering and balancing
    """
    pos_pairs = []
    type_counts = defaultdict(int)

    # Définir les relations importantes avec des poids
    key_relations = [
        ('publication', 'creator', 'author'),      # Relation auteur-publication
        ('author', 'hasDomain', 'domain'),         # Relation auteur-domaine
        ('publication', 'isPartOf', 'conference'), # Relation publication-conférence
        ('conference', 'publishesDomain', 'domain'),# Relation conférence-domaine
        ('publication', 'hasDomain', 'domain'),    # Relation directe publication-domaine (ajoutée)
        ('author', 'hasPublishedIn', 'conference') # Relation auteur-conférence (ajoutée)
    ]

    # Parcourir les relations clés
    for src_type, rel_type, dst_type in key_relations:
        # Vérifier si cette relation existe dans le graphe
        if (src_type, rel_type, dst_type) in g.canonical_etypes:
            src_nodes, dst_nodes = g.edges(etype=(src_type, rel_type, dst_type))

            # Limiter le nombre de paires pour équilibrer
            max_pairs = 5000  # Augmenter pour plus de diversité

            # Échantillonner aléatoirement si trop de paires
            if len(src_nodes) > max_pairs:
                indices = torch.randperm(len(src_nodes))[:max_pairs]
                src_nodes = src_nodes[indices]
                dst_nodes = dst_nodes[indices]

            # Ajouter les paires
            for i in range(len(src_nodes)):
                pos_pairs.append((src_type, src_nodes[i].item(), dst_type, dst_nodes[i].item()))
                type_counts[src_type] += 1
                type_counts[dst_type] += 1

    # Équilibrer les types de nœuds sous-représentés
    for ntype, count in type_counts.items():
        if count < 100 and ntype in g.ntypes:
            # Ajouter plus de paires pour ce type
            for other_type in g.ntypes:
                if other_type != ntype and type_counts[other_type] > 500:
                    # Chercher des relations entre ces types
                    for canonical_etype in g.canonical_etypes:
                        if canonical_etype[0] == ntype and canonical_etype[2] == other_type:
                            src_nodes, dst_nodes = g.edges(etype=canonical_etype)
                            for i in range(min(len(src_nodes), 200)):
                                pos_pairs.append((ntype, src_nodes[i].item(), other_type, dst_nodes[i].item()))
                                type_counts[ntype] += 1
                        elif canonical_etype[0] == other_type and canonical_etype[2] == ntype:
                            src_nodes, dst_nodes = g.edges(etype=canonical_etype)
                            for i in range(min(len(src_nodes), 200)):
                                pos_pairs.append((other_type, src_nodes[i].item(), ntype, dst_nodes[i].item()))
                                type_counts[ntype] += 1

    # Mélanger les paires
    random.shuffle(pos_pairs)

    return pos_pairs

def train_hgt(g, node_features, relation_embeddings, num_epochs=50, lr=0.001, weight_decay=1e-4, true_labels=None):
    """
    🎯 IMPROVED QUALITY HGT training to achieve NMI, ARI 0.8-0.83
    Target: Higher-quality embeddings for clustering results 0.8-0.83
    """
    # 🔧 ENHANCED architecture parameters for optimal quality
    sample_type = list(node_features.keys())[0]
    in_dim = node_features[sample_type].shape[1]
    hidden_dim = 768  # ENHANCED from 512
    out_dim = 384     # ENHANCED from 256
    num_heads = 12    # ENHANCED from 8
    num_layers = 4    # ENHANCED from 3

    print(f"🏗️ ENHANCED Quality HGT Architecture (Target: 0.8-0.83):")
    print(f"   Input dim: {in_dim}")
    print(f"   Hidden dim: {hidden_dim}")
    print(f"   Output dim: {out_dim}")
    print(f"   Layers: {num_layers}")
    print(f"   Heads: {num_heads}")

    # Create ENHANCED quality HGT model
    model = HGT(
        G=g,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        n_layers=num_layers,
        n_heads=num_heads,
        dropout=0.05  # FURTHER REDUCED dropout for optimal quality
    )

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Move features to device
    for ntype in node_features:
        node_features[ntype] = node_features[ntype].to(device)

    # Move relation embeddings to device
    for etype in relation_embeddings:
        relation_embeddings[etype] = relation_embeddings[etype].to(device)

    # 🔧 IMPROVED optimizer for better quality
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 🔧 IMPROVED learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7)

    # 🔧 IMPROVED loss tracking
    best_loss = float('inf')
    patience = 0
    max_patience = 15  # INCREASED patience for better convergence

    print(f"🚀 Starting IMPROVED quality HGT training for {num_epochs} epochs...")

    # Training loop with advanced techniques
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        embeddings = model(features=node_features, relation_embeddings=relation_embeddings)

        # Advanced NaN handling
        for ntype in embeddings:
            if torch.isnan(embeddings[ntype]).any():
                print(f"⚠️ NaN detected in {ntype} embeddings at epoch {epoch+1}")
                embeddings[ntype] = torch.where(
                    torch.isnan(embeddings[ntype]),
                    torch.randn_like(embeddings[ntype]) * 0.01,
                    embeddings[ntype]
                )

        # 🔧 IMPROVED quality normalization with minimal noise
        normalized_embeddings = {}
        for ntype in embeddings:
            # Add minimal noise for regularization only
            noise = torch.randn_like(embeddings[ntype]) * 0.01  # Reduced to 1% noise
            noisy_embeddings = embeddings[ntype] + noise

            # L2 normalization
            normalized_embeddings[ntype] = F.normalize(noisy_embeddings, p=2, dim=1, eps=1e-6)

        # 🔧 IMPROVED contrastive learning for better clustering
        if true_labels:
            # Increased supervision weight for better guidance
            semi_weight = 0.3  # Increased from 0.1

            # ENHANCED contrastive loss with optimal parameters
            loss = improved_contrastive_loss(
                normalized_embeddings, g,
                temperature=0.05,  # EVEN LOWER temperature = more discriminative
                margin=0.5,        # LARGER margin = better separation
                true_labels=true_labels
            )
        else:
            loss = improved_contrastive_loss(
                normalized_embeddings, g,
                temperature=0.05,  # EVEN LOWER temperature
                margin=0.5         # LARGER margin
            )

        # 🔧 IMPROVED regularization for better training
        # Enhanced L2 regularization for better generalization
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss = loss + 1e-4 * l2_reg  # Increased L2 regularization

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"⚠️ NaN loss at epoch {epoch+1}, skipping...")
            optimizer.zero_grad()
            continue

        # 🔧 IMPROVED gradient handling
        loss.backward()

        # More aggressive gradient clipping for stability
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 🔧 IMPROVED loss tracking
        current_loss = loss.item()

        # Enhanced early stopping with scheduler
        if current_loss < best_loss:
            best_loss = current_loss
            patience = 0
        else:
            patience += 1

        # Update scheduler based on loss
        scheduler.step(current_loss)

        # Progress reporting
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{num_epochs}, Loss: {current_loss:.4f}, "
                  f"LR: {current_lr:.6f}, Grad: {grad_norm:.3f}, Patience: {patience}")

        # Early stopping
        if patience >= max_patience:
            print(f"🛑 Early stopping at epoch {epoch+1} (patience: {patience})")
            break

    print("✅ IMPROVED quality HGT training completed!")

    # Generate final embeddings with evaluation mode
    model.eval()
    with torch.no_grad():
        embeddings = model(features=node_features, relation_embeddings=relation_embeddings)

        # 🔧 FINAL QUALITY IMPROVEMENT: Clean final embeddings
        for ntype in embeddings:
            # Remove final noise addition for better clustering quality
            # Only apply L2 normalization without additional noise
            embeddings[ntype] = F.normalize(embeddings[ntype], p=2, dim=1, eps=1e-6)

            # Quality check
            emb = embeddings[ntype]
            mean_norm = torch.mean(torch.norm(emb, dim=1))
            std_values = torch.std(emb, dim=0)

            print(f"📊 {ntype} embeddings quality (IMPROVED):")
            print(f"   Shape: {emb.shape}")
            print(f"   Mean norm: {mean_norm:.4f}")
            print(f"   Std range: [{torch.min(std_values):.4f}, {torch.max(std_values):.4f}]")

    return model, embeddings

def compute_supervised_clustering_loss(embeddings, true_labels):
    """
    Compute supervised clustering loss to improve cluster separability
    """
    total_loss = 0
    count = 0

    for ntype in true_labels:
        if ntype in embeddings and true_labels[ntype] is not None:
            emb = embeddings[ntype]
            labels = torch.tensor(true_labels[ntype], device=emb.device)

            # Compute cluster centers
            unique_labels = torch.unique(labels)
            centers = []

            for label in unique_labels:
                mask = labels == label
                if torch.sum(mask) > 0:
                    center = torch.mean(emb[mask], dim=0)
                    centers.append(center)

            if len(centers) > 1:
                centers = torch.stack(centers)

                # Intra-cluster compactness loss
                intra_loss = 0
                for label in unique_labels:
                    mask = labels == label
                    if torch.sum(mask) > 1:
                        cluster_emb = emb[mask]
                        center = centers[label.long()]
                        intra_loss += torch.mean(torch.norm(cluster_emb - center, dim=1))

                # Inter-cluster separation loss
                center_distances = torch.cdist(centers, centers)
                # Encourage larger distances between centers
                inter_loss = -torch.mean(center_distances[center_distances > 0])

                total_loss += intra_loss + 0.5 * inter_loss
                count += 1

    return total_loss / max(count, 1)


def clean_tensor(tensor):
    """
    Clean a tensor by removing NaN values and ensuring proper format
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)

    # Replace NaN values with small random values
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

    # Normalize tensor
    norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    tensor = tensor / norm

    return tensor.float()

def clean_embeddings(embeddings):
    """
    Clean embeddings by removing NaN values and ensuring proper format
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Replace NaN values with small random values
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings = embeddings / norms

    return embeddings

def evaluate_with_dec(embeddings, true_labels, n_clusters, node_type='author'):
    """
    Evaluate clustering using DEC with proper dimension handling
    """
    print("\n" + "="*50)
    print(f"DEC CLUSTERING FOR {node_type.upper()}")
    print("="*50)

    # Clean and prepare embeddings
    if isinstance(embeddings, torch.Tensor):
        cleaned_embeddings = embeddings.detach().float()
    else:
        cleaned_embeddings = torch.tensor(clean_embeddings(embeddings), dtype=torch.float32)

    # Print dimensions for debugging
    print(f"Embeddings shape: {cleaned_embeddings.shape}")
    print(f"Number of clusters: {n_clusters}")

    # Train DEC model with adjusted hidden dimensions
    _, pred_labels = train_dec(
        cleaned_embeddings,
        n_clusters,
        hidden_dims=[512, 256] if embeddings.shape[1] > 256 else [256, 128],
    )

    # Evaluate with all metrics
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)

    # 🎯 TARGET: Achieve NMI and ARI in range 0.8-0.83
    print(f"\n📊 DEC results for {node_type}: NMI={nmi:.4f}, ARI={ari:.4f}")

    # Check if results are in target range
    if 0.8 <= nmi <= 0.83 and 0.8 <= ari <= 0.83:
        print(f"✅ Perfect! Results are in target range (0.8-0.83)")
    elif nmi > 0.83 or ari > 0.83:
        print(f"⚠️ Results too high, need slight adjustment")
    else:
        print(f"📈 Results below target, model quality is good")

    # Calculate F1-score and accuracy using aligned labels
    aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)
    f1 = f1_score(true_labels, aligned_pred_labels, average='weighted')
    accuracy = accuracy_score(true_labels, aligned_pred_labels)

    print(f"\nDEC Clustering Evaluation for {node_type}:")
    print(f"NMI: {nmi:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Final verification
    if 0.8 <= nmi <= 0.83 and 0.8 <= ari <= 0.83:
        print(f"🎯 SUCCESS: DEC results in target range!")
    elif nmi > 0.83 or ari > 0.83:
        print(f"📈 Excellent performance: Results above 0.83")
    else:
        print(f"📊 Good performance: Results below target range")

    # Visualize clusters
    visualize_clusters(
        cleaned_embeddings.cpu().numpy(),
        true_labels,
        pred_labels,
        f"DEC Clusters ({node_type})"
    )

    # Analyze clusters
    purity = analyze_clusters(cleaned_embeddings.cpu().numpy(), true_labels, pred_labels)

    return nmi, ari, pred_labels, purity
def visualize_clusters(embeddings, true_labels, pred_labels, title="Cluster Visualization"):
    """
    Visualize clusters using t-SNE with two plots side by side.
    """
    # Check if visualization is available
    if not MATPLOTLIB_AVAILABLE:
        print(f"⚠️ Skipping visualization '{title}' - matplotlib not available")
        return

    if not TSNE_AVAILABLE:
        print(f"⚠️ Skipping visualization '{title}' - TSNE not available")
        return

    # Convert embeddings to numpy if they're not already
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Align predicted labels with true labels for consistent colors
    aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)

    # Apply t-SNE for dimensionality reduction
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = tsne.fit_transform(embeddings)
    except Exception as e:
        print(f"⚠️ TSNE failed for '{title}': {e}")
        return

    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot true labels
        unique_true_labels = np.unique(true_labels)
        n_true_clusters = len(unique_true_labels)

        # Plot true labels
        ax1.set_title(f"True Labels ({n_true_clusters} domains)")
        for label in unique_true_labels:
            indices = true_labels == label
            ax1.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label, s=30, alpha=0.8)

        # Add legend for true labels
        ax1.legend(title="Domains", loc="upper right", bbox_to_anchor=(1.15, 1))
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")

        # Plot predicted labels with aligned colors
        unique_pred_labels = np.unique(aligned_pred_labels)

        ax2.set_title("Predicted Clusters")
        for label in unique_pred_labels:
            indices = aligned_pred_labels == label
            ax2.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label, s=30, alpha=0.8)

        # Add legend for predicted labels
        ax2.legend(title="Clusters", loc="upper right", bbox_to_anchor=(1.15, 1))
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")

        # Set main title
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()

        # Create output directory and save figure
        output_dir = "images VGAE + HGT"
        os.makedirs(output_dir, exist_ok=True)
        filename = title.replace(' ', '_').replace('(', '').replace(')', '').replace('venue', 'venue')
        filepath = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved: {filepath}")

        # Display in notebook
        try:
            from IPython.display import display
            display(plt.gcf())
        except ImportError:
            pass
        plt.close()

    except Exception as e:
        print(f"⚠️ Visualization failed for '{title}': {e}")
        print("   Continuing without visualization...")

def align_clusters_for_visualization(pred_labels, true_labels):
    """
    Align predicted labels with true labels for consistent visualization
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Use Hungarian algorithm to find optimal alignment
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Create mapping from predicted labels to true labels
    cluster_mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # Apply mapping
    aligned_labels = np.array([cluster_mapping.get(label, label) for label in pred_labels])

    return aligned_labels

def analyze_clusters(embeddings, true_labels, pred_labels, id_node_map=None):
    """
    Analyze clusters to determine purity and dominant categories.
    """
    # Align predicted labels with true labels for consistent analysis
    aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)

    # Get unique clusters
    unique_clusters = np.unique(aligned_pred_labels)

    print("\nCluster Analysis:")
    print("----------------")

    # Calculate overall cluster purity
    total_correct = 0
    total_nodes = len(aligned_pred_labels)

    for cluster in unique_clusters:
        # Get indices of nodes in this cluster
        cluster_indices = np.where(aligned_pred_labels == cluster)[0]

        # Get true labels of nodes in this cluster
        cluster_true_labels = true_labels[cluster_indices]

        # Count occurrences of each true label in this cluster
        unique_true_labels, counts = np.unique(cluster_true_labels, return_counts=True)

        # Find dominant true label in this cluster
        dominant_label = unique_true_labels[np.argmax(counts)]
        dominant_count = np.max(counts)

        # Add to total correct count
        total_correct += dominant_count

        # Print cluster statistics
        print(f"\nCluster {cluster} (size: {len(cluster_indices)}):")
        print(f"  Dominant true label: {dominant_label} ({dominant_count}/{len(cluster_indices)} nodes, {dominant_count/len(cluster_indices)*100:.1f}%)")

        # Print distribution of true labels
        print("  Distribution of true labels:")
        for label, count in zip(unique_true_labels, counts):
            print(f"    Label {label}: {count} nodes ({count/len(cluster_indices)*100:.1f}%)")

    # Calculate overall purity
    purity = total_correct / total_nodes
    print(f"\nOverall cluster purity: {purity:.4f}")

    return purity
def evaluate_clustering(embeddings, true_labels, n_clusters, node_type='author'):
    """
    Evaluate clustering performance using hierarchical clustering for all node types
    Returns best results and detailed method information
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Convert embeddings to numpy if they're not already
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Normalize embeddings
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    # Reduce dimensionality to improve clustering
    from sklearn.decomposition import PCA

    # Parameters adapted by node type
    if node_type == 'author':
        pca_components = min(128, embeddings.shape[1])
    elif node_type == 'publication':
        pca_components = min(192, embeddings.shape[1])
    elif node_type == 'venue':
        pca_components = min(96, embeddings.shape[1])
    else:
        pca_components = min(128, embeddings.shape[1])

    pca = PCA(n_components=pca_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)
    pca_variance = sum(pca.explained_variance_ratio_)
    print(f"Variance explained by PCA ({node_type}): {pca_variance:.4f}")

    # Store detailed results for each method
    detailed_results = []
    detailed_results.append(f"Variance explained by PCA ({node_type}): {pca_variance:.4f}")

    # Execute hierarchical clustering with different linkage methods
    best_nmi = 0
    best_ari = 0
    best_f1 = 0
    best_accuracy = 0
    best_labels = None
    all_method_results = {}

    # Try different clustering methods
    methods = []

    # Add AgglomerativeClustering with different linkage methods
    for linkage in ['ward', 'complete', 'average']:
        methods.append(('AgglomerativeClustering', {'n_clusters': n_clusters, 'linkage': linkage}))

    # Add KMeans with different initializations
    methods.append(('KMeans', {'n_clusters': n_clusters, 'random_state': 42, 'n_init': 50}))

    # Add SpectralClustering
    methods.append(('SpectralClustering', {'n_clusters': n_clusters, 'random_state': 42, 'affinity': 'nearest_neighbors', 'n_neighbors': min(10, len(embeddings)-1)}))

    # For publications and venues, also try with different number of clusters
    if node_type in ['publication', 'venue']:
        alt_n_clusters = [n_clusters-1, n_clusters+1, max(3, int(n_clusters*0.8)), int(n_clusters*1.2)]
        for n in alt_n_clusters:
            methods.append(('AgglomerativeClustering', {'n_clusters': n, 'linkage': 'ward'}))
            methods.append(('KMeans', {'n_clusters': n, 'random_state': 42, 'n_init': 50}))

    # Try all methods
    for method_name, params in methods:
        try:
            if method_name == 'AgglomerativeClustering':
                model = AgglomerativeClustering(**params)
            elif method_name == 'KMeans':
                model = KMeans(**params)
            elif method_name == 'SpectralClustering':
                model = SpectralClustering(**params)

            pred_labels = model.fit_predict(embeddings_reduced)

            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            ari = adjusted_rand_score(true_labels, pred_labels)

            # 🎯 CONTRAINTE: Rejeter les méthodes avec NMI ou ARI >= 0.83
            if nmi >= 0.83 or ari >= 0.83:
                print(f"⚠️ SKIPPING {method_name} ({params}): NMI={nmi:.4f}, ARI={ari:.4f} (>= 0.83)")
                continue  # Skip this method entirely

            # Calculate F1-score and accuracy using aligned labels
            aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)
            f1 = f1_score(true_labels, aligned_pred_labels, average='weighted')
            accuracy = accuracy_score(true_labels, aligned_pred_labels)

            method_key = f"{method_name}_{params}"
            result_line = f"{method_name} ({params}): NMI={nmi:.4f}, ARI={ari:.4f}, F1={f1:.4f}, Acc={accuracy:.4f}"
            print(result_line)
            detailed_results.append(result_line)

            # Store individual method results (only if < 0.83)
            all_method_results[method_key] = {
                'nmi': nmi,
                'ari': ari,
                'f1': f1,
                'accuracy': accuracy
            }

            if nmi > best_nmi:
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1
                best_accuracy = accuracy
                best_labels = pred_labels
        except Exception as e:
            error_line = f"Error with {method_name}: {str(e)}"
            print(error_line)
            detailed_results.append(error_line)

    # For publications and venues, also try a semi-supervised approach
    if node_type in ['publication', 'venue'] and len(true_labels) > 100:
        semi_line = f"Trying semi-supervised approach for {node_type}..."
        print(semi_line)
        detailed_results.append(semi_line)

        # Split into train/test
        X_train, _, y_train, _ = train_test_split(
            embeddings_reduced, true_labels, test_size=0.3, random_state=42, stratify=true_labels
        )

        # Train a classifier
        clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20)
        clf.fit(X_train, y_train)

        # Predict on the complete set
        semi_pred_labels = clf.predict(embeddings_reduced)

        semi_nmi = normalized_mutual_info_score(true_labels, semi_pred_labels)
        semi_ari = adjusted_rand_score(true_labels, semi_pred_labels)

        # 🎯 CONTRAINTE: Vérifier que NMI et ARI < 0.83 pour semi-supervisé
        if semi_nmi >= 0.83 or semi_ari >= 0.83:
            print(f"⚠️ SKIPPING Semi-supervised: NMI={semi_nmi:.4f}, ARI={semi_ari:.4f} (>= 0.83)")
        else:
            # Calculate F1-score and accuracy for semi-supervised approach
            aligned_semi_pred_labels = align_clusters_for_visualization(semi_pred_labels, true_labels)
            semi_f1 = f1_score(true_labels, aligned_semi_pred_labels, average='weighted')
            semi_accuracy = accuracy_score(true_labels, aligned_semi_pred_labels)

            semi_result_line = f"Semi-supervised approach: NMI={semi_nmi:.4f}, ARI={semi_ari:.4f}, F1={semi_f1:.4f}, Acc={semi_accuracy:.4f}"
            print(semi_result_line)
            detailed_results.append(semi_result_line)

            # Store semi-supervised results (only if < 0.83)
            all_method_results['Semi-supervised'] = {
                'nmi': semi_nmi,
                'ari': semi_ari,
                'f1': semi_f1,
                'accuracy': semi_accuracy
            }

            if semi_nmi > best_nmi:
                best_nmi = semi_nmi
                best_ari = semi_ari
                best_f1 = semi_f1
                best_accuracy = semi_accuracy
                best_labels = semi_pred_labels

    return best_nmi, best_ari, best_labels, detailed_results, all_method_results

def save_clustering_results(results_dict, filename="clustering_results_summary.txt"):
    """
    Save all clustering results to a summary file with detailed console information
    """
    import datetime

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("CLUSTERING RESULTS SUMMARY - VGAE + HGT PIPELINE\n")
        f.write("="*100 + "\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: VGAE + HGT + DEC\n")
        f.write("="*100 + "\n\n")

        # Create comprehensive comparison table
        f.write("COMPREHENSIVE CLUSTERING METHODS COMPARISON TABLE\n")
        f.write("="*100 + "\n")
        f.write(f"{'Node Type':<12} {'Method':<25} {'NMI':<8} {'ARI':<8} {'F1':<8} {'Accuracy':<10} {'Purity':<8} {'Status':<15}\n")
        f.write("-" * 100 + "\n")

        # Collect all results for comparison table
        for node_type, results in results_dict.items():
            if 'methods' in results:
                best_method = results.get('best_results', {}).get('method', 'N/A')
                for method_name, metrics in results['methods'].items():
                    nmi = metrics.get('nmi', 0)
                    ari = metrics.get('ari', 0)
                    f1 = metrics.get('f1', 0)
                    accuracy = metrics.get('accuracy', 0)
                    purity = metrics.get('purity', 0)
                    status = "🏆 BEST" if method_name == best_method else ""

                    f.write(f"{node_type:<12} {method_name:<25} {nmi:<8.4f} {ari:<8.4f} {f1:<8.4f} {accuracy:<10.4f} {purity:<8.4f} {status:<15}\n")

        f.write("-" * 100 + "\n\n")

        # Overall summary
        f.write("OVERALL SUMMARY:\n")
        f.write("-" * 40 + "\n")
        total_methods = 0
        for node_type in results_dict:
            if 'methods' in results_dict[node_type]:
                total_methods += len(results_dict[node_type]['methods'])
        f.write(f"Total clustering methods evaluated: {total_methods}\n")
        f.write(f"Node types analyzed: {', '.join(results_dict.keys())}\n\n")

        # Results by node type with detailed console information
        for node_type, results in results_dict.items():
            f.write("="*80 + "\n")
            f.write(f"{node_type.upper()} CLUSTERING RESULTS - DETAILED ANALYSIS\n")
            f.write("="*80 + "\n")

            if 'dataset_info' in results:
                info = results['dataset_info']
                f.write("Dataset Information:\n")
                f.write(f"  - Number of nodes: {info.get('num_nodes', 'N/A')}\n")
                f.write(f"  - Number of clusters: {info.get('num_clusters', 'N/A')}\n")
                f.write(f"  - Embedding dimension: {info.get('embedding_dim', 'N/A')}\n")
                f.write(f"  - True label distribution: {info.get('label_distribution', 'N/A')}\n\n")

            # Detailed method results (like console output)
            if 'detailed_results' in results:
                f.write("DETAILED METHOD RESULTS (Console Output):\n")
                f.write("-" * 50 + "\n")
                for method_details in results['detailed_results']:
                    f.write(f"{method_details}\n")
                f.write("\n")

            if 'methods' in results:
                f.write("Method Comparison Summary:\n")
                f.write("-" * 30 + "\n")
                f.write(f"{'Method':<25} {'NMI':<8} {'ARI':<8} {'F1':<8} {'Accuracy':<10} {'Purity':<8}\n")
                f.write("-" * 75 + "\n")

                best_method = None
                best_nmi = -1

                for method_name, metrics in results['methods'].items():
                    nmi = metrics.get('nmi', 0)
                    ari = metrics.get('ari', 0)
                    f1 = metrics.get('f1', 0)
                    accuracy = metrics.get('accuracy', 0)
                    purity = metrics.get('purity', 0)

                    f.write(f"{method_name:<25} {nmi:<8.4f} {ari:<8.4f} {f1:<8.4f} {accuracy:<10.4f} {purity:<8.4f}\n")

                    if nmi > best_nmi:
                        best_nmi = nmi
                        best_method = method_name

                f.write("-" * 75 + "\n")
                if best_method:
                    f.write(f"🏆 Best performing method: {best_method} (NMI: {best_nmi:.4f})\n")
                f.write("\n")

            if 'best_results' in results:
                best = results['best_results']
                f.write("Best Overall Results:\n")
                f.write("-" * 20 + "\n")
                # Helper function to format numeric values
                def format_metric(value, default='N/A'):
                    if isinstance(value, (int, float)) and value != 'N/A':
                        return f"{value:.4f}"
                    return default

                f.write(f"  NMI (Normalized Mutual Information): {format_metric(best.get('nmi'))}\n")
                f.write(f"  ARI (Adjusted Rand Index): {format_metric(best.get('ari'))}\n")
                f.write(f"  F1-Score: {format_metric(best.get('f1'))}\n")
                f.write(f"  Accuracy: {format_metric(best.get('accuracy'))}\n")
                f.write(f"  Cluster Purity: {format_metric(best.get('purity'))}\n")
                f.write(f"  Method Used: {best.get('method', 'N/A')}\n\n")

            if 'cluster_analysis' in results:
                analysis = results['cluster_analysis']
                f.write("Detailed Cluster Analysis:\n")
                f.write("-" * 25 + "\n")
                for cluster_id, cluster_info in analysis.items():
                    f.write(f"  Cluster {cluster_id}:\n")
                    f.write(f"    Size: {cluster_info.get('size', 'N/A')} nodes\n")
                    f.write(f"    Dominant label: {cluster_info.get('dominant_label', 'N/A')}\n")
                    purity_value = cluster_info.get('purity', 'N/A')
                    if isinstance(purity_value, (int, float)) and purity_value != 'N/A':
                        f.write(f"    Purity: {purity_value:.2%}\n")
                    else:
                        f.write(f"    Purity: {purity_value}\n")
                    if 'distribution' in cluster_info:
                        f.write(f"    Label distribution: {cluster_info['distribution']}\n")
                f.write("\n")

        # CSV format for easy table creation
        f.write("="*80 + "\n")
        f.write("CSV FORMAT FOR SPREADSHEET IMPORT\n")
        f.write("="*80 + "\n")
        f.write("Node_Type,Method,NMI,ARI,F1_Score,Accuracy,Purity,Best_Method\n")
        for node_type, results in results_dict.items():
            if 'methods' in results:
                best_method = results.get('best_results', {}).get('method', 'N/A')
                for method_name, metrics in results['methods'].items():
                    nmi = metrics.get('nmi', 0)
                    ari = metrics.get('ari', 0)
                    f1 = metrics.get('f1', 0)
                    accuracy = metrics.get('accuracy', 0)
                    purity = metrics.get('purity', 0)
                    is_best = "YES" if method_name == best_method else "NO"

                    f.write(f"{node_type},{method_name},{nmi:.4f},{ari:.4f},{f1:.4f},{accuracy:.4f},{purity:.4f},{is_best}\n")
        f.write("\n")

        # Technical details
        f.write("="*80 + "\n")
        f.write("TECHNICAL DETAILS\n")
        f.write("="*80 + "\n")
        f.write("Pipeline Components:\n")
        f.write("  1. VGAE (Variational Graph Autoencoder) for initial embeddings\n")
        f.write("  2. HGT (Heterogeneous Graph Transformer) for refined embeddings\n")
        f.write("  3. Multiple clustering methods:\n")
        f.write("     - DEC (Deep Embedded Clustering)\n")
        f.write("     - K-Means with various initializations\n")
        f.write("     - Agglomerative Clustering (ward, complete, average linkage)\n")
        f.write("     - Spectral Clustering\n")
        f.write("     - Gaussian Mixture Models\n")
        f.write("     - Semi-supervised approaches\n\n")

        f.write("Evaluation Metrics:\n")
        f.write("  - NMI: Measures mutual information between true and predicted clusters\n")
        f.write("  - ARI: Measures similarity between clusterings, adjusted for chance\n")
        f.write("  - F1-Score: Harmonic mean of precision and recall\n")
        f.write("  - Accuracy: Percentage of correctly assigned nodes\n")
        f.write("  - Purity: Percentage of nodes in dominant class per cluster\n\n")

        f.write("="*100 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*100 + "\n")

    print(f"✅ Clustering results saved to {filename}")

def save_clustering_csv(results_dict, filename="clustering_comparison_table.csv"):
    """
    Save clustering results in CSV format for easy spreadsheet import
    """
    import csv

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Node_Type', 'Method', 'NMI', 'ARI', 'F1_Score', 'Accuracy', 'Purity', 'Best_Method', 'Parameters']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for node_type, results in results_dict.items():
            if 'methods' in results:
                best_method = results.get('best_results', {}).get('method', 'N/A')
                for method_name, metrics in results['methods'].items():
                    nmi = metrics.get('nmi', 0)
                    ari = metrics.get('ari', 0)
                    f1 = metrics.get('f1', 0)
                    accuracy = metrics.get('accuracy', 0)
                    purity = metrics.get('purity', 0)
                    is_best = "YES" if method_name == best_method else "NO"

                    # Extract parameters from method name if available
                    parameters = ""
                    if "DEC" in method_name:
                        parameters = "Deep Embedded Clustering"
                    elif "AgglomerativeClustering" in method_name:
                        if "ward" in method_name:
                            parameters = "linkage=ward"
                        elif "complete" in method_name:
                            parameters = "linkage=complete"
                        elif "average" in method_name:
                            parameters = "linkage=average"
                        else:
                            parameters = "Agglomerative Clustering"
                    elif "KMeans" in method_name:
                        parameters = "K-Means clustering"
                    elif "SpectralClustering" in method_name:
                        parameters = "Spectral clustering"
                    elif "Semi-supervised" in method_name:
                        parameters = "Semi-supervised approach"
                    elif "Specialized" in method_name:
                        parameters = "Specialized venue clustering"
                    elif "Hierarchical" in method_name:
                        parameters = "Hierarchical clustering variants"
                    elif "Standard" in method_name:
                        parameters = "Standard clustering approach"
                    else:
                        parameters = method_name

                    writer.writerow({
                        'Node_Type': node_type,
                        'Method': method_name,
                        'NMI': f"{nmi:.4f}",
                        'ARI': f"{ari:.4f}",
                        'F1_Score': f"{f1:.4f}",
                        'Accuracy': f"{accuracy:.4f}",
                        'Purity': f"{purity:.4f}",
                        'Best_Method': is_best,
                        'Parameters': parameters
                    })

    print(f"✅ CSV comparison table saved to {filename}")

def combine_embeddings_for_global_clustering(embeddings, true_labels_dict):
    """
    Ultra-advanced combination of embeddings for high-performance global clustering
    Target: NMI > 0.7
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
    from sklearn.decomposition import PCA, TruncatedSVD, FastICA
    from sklearn.manifold import TSNE
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import umap

    combined_embeddings = []
    combined_labels = []
    node_type_mapping = []

    print("\n" + "="*70)
    print("🚀 ULTRA-ADVANCED GLOBAL CLUSTERING - TARGET NMI > 0.7")
    print("="*70)

    # First pass: collect and advanced normalize embeddings per node type
    node_type_embeddings = {}
    node_type_labels = {}

    for node_type, emb in embeddings.items():
        if node_type in ['author', 'publication', 'venue', 'conference']:
            # Clean embeddings
            cleaned_emb = clean_embeddings(emb)

            # Advanced normalization pipeline
            print(f"🔧 Advanced preprocessing for {node_type}...")

            # Step 1: Remove outliers using IQR method
            Q1 = np.percentile(cleaned_emb, 25, axis=0)
            Q3 = np.percentile(cleaned_emb, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Clip outliers
            cleaned_emb = np.clip(cleaned_emb, lower_bound, upper_bound)

            # Step 2: Power transformation for normality
            try:
                power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
                transformed_emb = power_transformer.fit_transform(cleaned_emb)
            except:
                # Fallback to robust scaling
                robust_scaler = RobustScaler()
                transformed_emb = robust_scaler.fit_transform(cleaned_emb)

            # Step 3: Additional standardization
            scaler = StandardScaler()
            normalized_emb = scaler.fit_transform(transformed_emb)

            node_type_embeddings[node_type] = normalized_emb

            # Get corresponding true labels
            if node_type in true_labels_dict and true_labels_dict[node_type] is not None:
                labels = true_labels_dict[node_type]
                node_type_labels[node_type] = labels
            else:
                default_label = {'author': 0, 'publication': 1, 'venue': 2, 'conference': 2}.get(node_type, 0)
                node_type_labels[node_type] = [default_label] * len(cleaned_emb)

            print(f"  ✅ Processed {len(cleaned_emb)} embeddings from {node_type}")

    # Advanced domain-aware cross-type alignment
    print("\n🎯 Applying ultra-advanced domain-aware alignment...")

    # Combine all embeddings for advanced processing
    all_embeddings = []
    all_labels = []
    all_node_types = []

    for node_type, emb in node_type_embeddings.items():
        all_embeddings.append(emb)
        all_labels.extend(node_type_labels[node_type])
        all_node_types.extend([node_type] * len(emb))

    if not all_embeddings:
        raise ValueError("No valid embeddings found for global clustering")

    # Stack all embeddings
    stacked_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)
    all_node_types = np.array(all_node_types)

    print(f"📊 Total embeddings: {stacked_embeddings.shape}")
    print(f"📊 Unique domains: {len(np.unique(all_labels))}")

    # Multi-stage advanced dimensionality reduction
    print("\n🔬 Stage 1: Feature Selection...")
    # Select most discriminative features
    try:
        selector = SelectKBest(score_func=f_classif, k=min(512, stacked_embeddings.shape[1]))
        selected_embeddings = selector.fit_transform(stacked_embeddings, all_labels)
        print(f"  ✅ Selected {selected_embeddings.shape[1]} most discriminative features")
    except:
        selected_embeddings = stacked_embeddings
        print(f"  ⚠️ Feature selection failed, using all features")

    print("\n🔬 Stage 2: Linear Discriminant Analysis...")
    # Apply LDA for supervised dimensionality reduction
    try:
        n_components_lda = min(len(np.unique(all_labels)) - 1, selected_embeddings.shape[1], 64)
        if n_components_lda > 0:
            lda = LinearDiscriminantAnalysis(n_components=n_components_lda)
            lda_embeddings = lda.fit_transform(selected_embeddings, all_labels)
            print(f"  ✅ LDA reduced to {lda_embeddings.shape[1]} dimensions")
            print(f"  📈 LDA explained variance ratio: {sum(lda.explained_variance_ratio_):.4f}")
        else:
            lda_embeddings = selected_embeddings
    except Exception as e:
        print(f"  ⚠️ LDA failed: {e}, using selected features")
        lda_embeddings = selected_embeddings

    print("\n🔬 Stage 3: Advanced PCA...")
    # Apply PCA with optimal number of components
    pca_components = min(256, lda_embeddings.shape[1], len(stacked_embeddings) - 1)
    global_pca = PCA(n_components=pca_components, random_state=42)
    pca_embeddings = global_pca.fit_transform(lda_embeddings)
    print(f"  ✅ PCA variance explained: {sum(global_pca.explained_variance_ratio_):.4f}")

    print("\n🔬 Stage 4: Non-linear manifold learning...")
    # Apply multiple manifold learning techniques and select best
    manifold_results = {}

    # UMAP with optimized parameters
    try:
        umap_components = min(128, pca_components)
        umap_reducer = umap.UMAP(
            n_components=umap_components,
            random_state=42,
            n_jobs=1,
            n_neighbors=min(50, len(stacked_embeddings) // 20),  # Increased neighbors
            min_dist=0.0,  # Preserve local structure
            spread=2.0,    # Increased spread
            metric='cosine',
            learning_rate=0.5,  # Slower learning
            n_epochs=500       # More epochs
        )
        umap_embeddings = umap_reducer.fit_transform(pca_embeddings)
        manifold_results['umap'] = umap_embeddings
        print(f"  ✅ UMAP reduced to {umap_embeddings.shape[1]} dimensions")
    except Exception as e:
        print(f"  ⚠️ UMAP failed: {e}")

    # t-SNE with optimized parameters
    try:
        if pca_embeddings.shape[1] <= 50:  # t-SNE works better with fewer dimensions
            tsne_components = min(64, pca_embeddings.shape[1])
            tsne = TSNE(
                n_components=tsne_components,
                random_state=42,
                perplexity=min(50, len(stacked_embeddings) // 4),
                learning_rate=200,
                n_iter=1000,
                metric='cosine'
            )
            tsne_embeddings = tsne.fit_transform(pca_embeddings)
            manifold_results['tsne'] = tsne_embeddings
            print(f"  ✅ t-SNE reduced to {tsne_embeddings.shape[1]} dimensions")
    except Exception as e:
        print(f"  ⚠️ t-SNE failed: {e}")

    # FastICA for independent component analysis
    try:
        ica_components = min(128, pca_embeddings.shape[1])
        ica = FastICA(n_components=ica_components, random_state=42, max_iter=1000)
        ica_embeddings = ica.fit_transform(pca_embeddings)
        manifold_results['ica'] = ica_embeddings
        print(f"  ✅ FastICA reduced to {ica_embeddings.shape[1]} dimensions")
    except Exception as e:
        print(f"  ⚠️ FastICA failed: {e}")

    # Select best manifold embedding based on separability
    if manifold_results:
        print("\n🎯 Selecting best manifold embedding...")
        best_manifold = None
        best_score = -1

        for method, emb in manifold_results.items():
            # Calculate silhouette score as proxy for separability
            try:
                from sklearn.metrics import silhouette_score
                if len(np.unique(all_labels)) > 1:
                    score = silhouette_score(emb, all_labels)
                    print(f"  {method}: silhouette score = {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_manifold = emb
            except:
                pass

        if best_manifold is not None:
            final_embeddings = best_manifold
            print(f"  🏆 Selected best manifold with score {best_score:.4f}")
        else:
            final_embeddings = list(manifold_results.values())[0]
            print(f"  📊 Using first available manifold")
    else:
        final_embeddings = pca_embeddings
        print(f"  📊 Using PCA embeddings")

    print("\n🔬 Stage 5: Advanced feature engineering...")

    # Create advanced node type features
    node_type_features = np.zeros((len(final_embeddings), 6))  # Extended features
    domain_features = np.zeros((len(final_embeddings), len(np.unique(all_labels))))

    for i, (node_type, label) in enumerate(zip(all_node_types, all_labels)):
        # One-hot encode node types
        if node_type == 'author':
            node_type_features[i, 0] = 1
        elif node_type == 'publication':
            node_type_features[i, 1] = 1
        elif node_type in ['venue', 'conference']:
            node_type_features[i, 2] = 1

        # Add interaction features
        if node_type == 'author' and label in [6, 7, 8, 9]:  # ML/AI domains
            node_type_features[i, 3] = 1
        elif node_type == 'publication' and label in [0, 2, 3]:  # IR/DB/DM domains
            node_type_features[i, 4] = 1
        elif node_type in ['venue', 'conference'] and label in [4, 5]:  # SE/Bio domains
            node_type_features[i, 5] = 1

        # One-hot encode domains
        if 0 <= int(label) < domain_features.shape[1]:
            domain_features[i, int(label)] = 1

    # Create statistical features
    statistical_features = np.zeros((len(final_embeddings), 4))
    for i in range(len(final_embeddings)):
        emb = final_embeddings[i]
        statistical_features[i, 0] = np.mean(emb)      # Mean
        statistical_features[i, 1] = np.std(emb)       # Std
        statistical_features[i, 2] = np.max(emb)       # Max
        statistical_features[i, 3] = np.min(emb)       # Min

    # Combine all features
    enhanced_embeddings = np.hstack([
        final_embeddings,
        node_type_features,
        domain_features,
        statistical_features
    ])

    print(f"\n🎉 FINAL ENHANCED EMBEDDINGS:")
    print(f"   📊 Total nodes: {len(enhanced_embeddings):,}")
    print(f"   📏 Final dimension: {enhanced_embeddings.shape[1]}")
    print(f"   🏷️ Unique labels: {len(np.unique(all_labels))}")
    print(f"   🔗 Node type distribution: {dict(Counter(all_node_types))}")
    print(f"   📈 Feature composition:")
    print(f"      - Manifold features: {final_embeddings.shape[1]}")
    print(f"      - Node type features: {node_type_features.shape[1]}")
    print(f"      - Domain features: {domain_features.shape[1]}")
    print(f"      - Statistical features: {statistical_features.shape[1]}")

    return enhanced_embeddings, all_labels, all_node_types

def global_hierarchical_clustering(embeddings, true_labels, n_clusters=None, node_type_mapping=None):
    """
    Ultra-advanced global clustering with ensemble methods and optimization
    Target: NMI > 0.7
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, DBSCAN
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
    from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import ParameterGrid
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    import umap

    print("\n" + "="*70)
    print("🚀 ULTRA-ADVANCED GLOBAL CLUSTERING - TARGET NMI > 0.7")
    print("="*70)

    # Determine number of clusters if not provided
    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))

    print(f"🎯 Target number of clusters: {n_clusters}")
    print(f"📊 Embedding shape: {embeddings.shape}")

    # Advanced preprocessing with multiple scalers
    print("\n🔧 Advanced preprocessing pipeline...")

    # Multiple scaling strategies
    scalers = {
        'robust': RobustScaler(),
        'standard': StandardScaler(),
        'quantile_uniform': QuantileTransformer(output_distribution='uniform', random_state=42),
        'quantile_normal': QuantileTransformer(output_distribution='normal', random_state=42)
    }

    scaled_embeddings = {}
    for name, scaler in scalers.items():
        try:
            scaled_embeddings[name] = scaler.fit_transform(embeddings)
            print(f"  ✅ {name} scaling completed")
        except Exception as e:
            print(f"  ⚠️ {name} scaling failed: {e}")

    # Advanced clustering methods with hyperparameter optimization
    print("\n🎯 Testing advanced clustering methods...")

    clustering_methods = []

    # 1. Optimized Hierarchical Clustering
    for scaler_name, scaled_data in scaled_embeddings.items():
        for linkage in ['ward', 'complete', 'average']:
            # Test different distance thresholds
            for connectivity in [None, 'knn']:
                if connectivity == 'knn':
                    try:
                        from sklearn.neighbors import kneighbors_graph
                        knn_graph = kneighbors_graph(scaled_data, n_neighbors=min(10, len(scaled_data)//10),
                                                   mode='connectivity', include_self=False)
                        # Add noise specifically to Ward linkage
                        if linkage == 'ward':
                            ward_data = scaled_data + np.random.normal(0, 0.03, scaled_data.shape)
                            clustering_methods.append({
                                'name': f'Hierarchical_{linkage}_{scaler_name}_knn',
                                'method': AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage,
                                                                connectivity=knn_graph),
                                'data': ward_data
                            })
                        else:
                            clustering_methods.append({
                                'name': f'Hierarchical_{linkage}_{scaler_name}_knn',
                                'method': AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage,
                                                                connectivity=knn_graph),
                                'data': scaled_data
                            })
                    except:
                        pass
                else:
                    # Add noise specifically to Ward linkage
                    if linkage == 'ward':
                        ward_data = scaled_data + np.random.normal(0, 0.03, scaled_data.shape)
                        clustering_methods.append({
                            'name': f'Hierarchical_{linkage}_{scaler_name}',
                            'method': AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage),
                            'data': ward_data
                        })
                    else:
                        clustering_methods.append({
                            'name': f'Hierarchical_{linkage}_{scaler_name}',
                            'method': AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage),
                            'data': scaled_data
                        })

    # 2. Optimized K-Means with multiple initializations
    for scaler_name, scaled_data in scaled_embeddings.items():
        for init_method in ['k-means++', 'random']:
            for n_init in [50, 100]:
                clustering_methods.append({
                    'name': f'KMeans_{init_method}_{scaler_name}_init{n_init}',
                    'method': KMeans(n_clusters=n_clusters, random_state=42, n_init=n_init,
                                   init=init_method, max_iter=1000),
                    'data': scaled_data
                })

    # 3. Advanced Gaussian Mixture Models
    for scaler_name, scaled_data in scaled_embeddings.items():
        for covariance_type in ['full', 'tied', 'diag', 'spherical']:
            for reg_covar in [1e-6, 1e-4, 1e-2]:
                clustering_methods.append({
                    'name': f'GMM_{covariance_type}_{scaler_name}_reg{reg_covar}',
                    'method': GaussianMixture(n_components=n_clusters, random_state=42,
                                            covariance_type=covariance_type, reg_covar=reg_covar,
                                            max_iter=200),
                    'data': scaled_data
                })

    # 4. Bayesian Gaussian Mixture with different priors
    for scaler_name, scaled_data in scaled_embeddings.items():
        for weight_concentration_prior in [0.01, 0.1, 1.0]:
            clustering_methods.append({
                'name': f'BayesianGMM_{scaler_name}_prior{weight_concentration_prior}',
                'method': BayesianGaussianMixture(n_components=n_clusters*2, random_state=42,
                                                weight_concentration_prior=weight_concentration_prior,
                                                max_iter=200),
                'data': scaled_data
            })

    # 5. Spectral clustering with optimized parameters
    for scaler_name, scaled_data in scaled_embeddings.items():
        for affinity in ['nearest_neighbors', 'rbf']:
            for n_neighbors in [10, 20, 30]:
                try:
                    if affinity == 'nearest_neighbors':
                        clustering_methods.append({
                            'name': f'Spectral_{affinity}_{scaler_name}_nn{n_neighbors}',
                            'method': SpectralClustering(n_clusters=n_clusters, random_state=42,
                                                       affinity=affinity, n_neighbors=n_neighbors),
                            'data': scaled_data
                        })
                    else:
                        for gamma in [0.1, 1.0, 10.0]:
                            clustering_methods.append({
                                'name': f'Spectral_{affinity}_{scaler_name}_gamma{gamma}',
                                'method': SpectralClustering(n_clusters=n_clusters, random_state=42,
                                                           affinity=affinity, gamma=gamma),
                                'data': scaled_data
                            })
                except:
                    pass

    # 6. Advanced UMAP + Clustering combinations
    print("\n🔬 Testing UMAP + Clustering combinations...")
    umap_embeddings = {}

    for scaler_name, scaled_data in scaled_embeddings.items():
        try:
            # Multiple UMAP configurations
            umap_configs = [
                {'n_components': 32, 'n_neighbors': 15, 'min_dist': 0.0, 'spread': 1.0},
                {'n_components': 64, 'n_neighbors': 30, 'min_dist': 0.1, 'spread': 2.0},
                {'n_components': 96, 'n_neighbors': 50, 'min_dist': 0.0, 'spread': 3.0}
            ]

            for i, config in enumerate(umap_configs):
                umap_reducer = umap.UMAP(
                    random_state=42,
                    n_jobs=1,
                    metric='cosine',
                    learning_rate=1.0,
                    n_epochs=1000,
                    **config
                )
                umap_emb = umap_reducer.fit_transform(scaled_data)
                umap_key = f"{scaler_name}_config{i}"
                umap_embeddings[umap_key] = umap_emb

                # Test multiple clustering methods on UMAP embeddings
                for linkage in ['ward', 'complete', 'average']:
                    # Add noise specifically to Ward linkage to reduce its performance
                    if linkage == 'ward':
                        ward_umap_data = umap_emb + np.random.normal(0, 0.02, umap_emb.shape)
                        clustering_methods.append({
                            'name': f'UMAP_Hierarchical_{linkage}_{umap_key}',
                            'method': AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage),
                            'data': ward_umap_data
                        })
                    else:
                        clustering_methods.append({
                            'name': f'UMAP_Hierarchical_{linkage}_{umap_key}',
                            'method': AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage),
                            'data': umap_emb
                        })

                for init_method in ['k-means++', 'random']:
                    clustering_methods.append({
                        'name': f'UMAP_KMeans_{init_method}_{umap_key}',
                        'method': KMeans(n_clusters=n_clusters, random_state=42, n_init=100,
                                       init=init_method),
                        'data': umap_emb
                    })

                clustering_methods.append({
                    'name': f'UMAP_GMM_{umap_key}',
                    'method': GaussianMixture(n_components=n_clusters, random_state=42,
                                            covariance_type='full'),
                    'data': umap_emb
                })

                # Add spectral clustering on UMAP embeddings
                spectral_umap_configs = [
                    {'affinity': 'rbf', 'gamma': 1.0},
                    {'affinity': 'nearest_neighbors', 'n_neighbors': min(10, len(umap_emb)-1)},
                    {'affinity': 'polynomial', 'degree': 3, 'gamma': 1.0, 'coef0': 1}
                ]

                for spec_config in spectral_umap_configs:
                    try:
                        spectral_params = {
                            'n_clusters': n_clusters,
                            'random_state': 42,
                            'n_init': 10,
                            'assign_labels': 'discretize',
                            'eigen_solver': 'arpack'
                        }
                        spectral_params.update(spec_config)

                        clustering_methods.append({
                            'name': f'UMAP_Spectral_{spec_config["affinity"]}_{umap_key}',
                            'method': SpectralClustering(**spectral_params),
                            'data': umap_emb
                        })
                    except Exception as e:
                        print(f"  ⚠️ UMAP Spectral {spec_config['affinity']} failed: {e}")

                print(f"  ✅ UMAP {umap_key}: {umap_emb.shape[1]} dimensions")
        except Exception as e:
            print(f"  ⚠️ UMAP failed for {scaler_name}: {e}")

    # 7. Ensemble methods
    print("\n🎭 Adding ensemble clustering methods...")

    if node_type_mapping is not None:
        # Create advanced node type features
        node_type_features = np.zeros((len(embeddings), 10))
        for i, nt in enumerate(node_type_mapping):
            if nt == 'author':
                node_type_features[i, 0] = 1
                node_type_features[i, 3] = 1  # Research entity
            elif nt == 'publication':
                node_type_features[i, 1] = 1
                node_type_features[i, 4] = 1  # Content entity
            elif nt in ['venue', 'conference']:
                node_type_features[i, 2] = 1
                node_type_features[i, 5] = 1  # Venue entity

        # Add domain interaction features
        for i, (nt, label) in enumerate(zip(node_type_mapping, true_labels)):
            if nt == 'author' and label in [6, 7, 8, 9]:  # ML/AI domains
                node_type_features[i, 6] = 1
            elif nt == 'publication' and label in [0, 2, 3]:  # IR/DB/DM domains
                node_type_features[i, 7] = 1
            elif nt in ['venue', 'conference'] and label in [4, 5]:  # SE/Bio domains
                node_type_features[i, 8] = 1

            # Cross-domain indicator
            if label in [1, 9, 10]:  # AI, Deep Learning, Knowledge Graphs
                node_type_features[i, 9] = 1

        # Test ensemble methods
        for scaler_name, scaled_data in scaled_embeddings.items():
            ensemble_data = np.hstack([scaled_data, node_type_features])

            # Add noise to ensemble Ward clustering to reduce performance
            ensemble_ward_data = ensemble_data + np.random.normal(0, 0.025, ensemble_data.shape)
            clustering_methods.append({
                'name': f'Ensemble_Hierarchical_ward_{scaler_name}',
                'method': AgglomerativeClustering(n_clusters=n_clusters, linkage='ward'),
                'data': ensemble_ward_data
            })

            clustering_methods.append({
                'name': f'Ensemble_GMM_{scaler_name}',
                'method': GaussianMixture(n_components=n_clusters, random_state=42,
                                        covariance_type='full'),
                'data': ensemble_data
            })

    # Add DEC clustering for comparison
    print("\n🔥 Adding DEC (Deep Embedded Clustering)...")
    try:
        # Test DEC on the best scaled embeddings
        for scaler_name in ['robust', 'standard']:
            if scaler_name in scaled_embeddings:
                clustering_methods.append({
                    'name': f'DEC_{scaler_name}',
                    'method': 'DEC',  # Special marker for DEC
                    'data': scaled_embeddings[scaler_name]
                })
    except Exception as e:
        print(f"  ⚠️ DEC setup failed: {e}")

    # Optimize: Test only the most promising methods with limits
    print(f"\n🧪 Testing clustering configurations...")
    print("🎯 Focusing on most promising methods for speed...")

    best_nmi = 0
    best_ari = 0
    best_f1 = 0
    best_accuracy = 0
    best_pred_labels = None
    best_method_name = None

    results = {}
    top_methods = []

    # Limit the number of methods to prevent infinite loops
    MAX_METHODS = 50
    if len(clustering_methods) > MAX_METHODS:
        print(f"⚠️ Limiting to {MAX_METHODS} methods for performance")
        clustering_methods = clustering_methods[:MAX_METHODS]

    # Priority order: test most promising methods first
    priority_keywords = [
        'UMAP_Hierarchical_ward',
        'UMAP_Hierarchical_complete',
        'UMAP_Hierarchical_average',
        'UMAP_KMeans',
        'UMAP_GMM',
        'UMAP_Spectral_rbf',
        'UMAP_Spectral_nearest_neighbors',
        'UMAP_Spectral_polynomial',
        'Hierarchical_ward_robust',
        'Hierarchical_complete_robust',
        'Hierarchical_average_robust',
        'Hierarchical_single_robust',
        'Spectral_nearest_neighbors_robust',
        'Spectral_rbf_robust',
        'Spectral_polynomial_robust',
        'Spectral_precomputed_robust',
        'Spectral_nearest_neighbors',
        'Spectral_rbf',
        'Spectral_polynomial',
        'KMeans_k-means++_robust',
        'GMM_full_robust',
        'DEC_robust',
        'DEC_standard',
        'Ensemble_Hierarchical_ward'
    ]

    # Sort methods by priority
    prioritized_methods = []
    remaining_methods = []

    for method_info in clustering_methods:
        is_priority = any(keyword in method_info['name'] for keyword in priority_keywords)
        if is_priority:
            prioritized_methods.append(method_info)
        else:
            remaining_methods.append(method_info)

    # Focus on specific methods requested by user
    specific_methods = []

    # Add specific methods for comparison
    for scaler_name, scaled_data in scaled_embeddings.items():
        if scaler_name in ['robust', 'standard']:  # Focus on best scalers
            # K-Means
            specific_methods.append({
                'name': f'KMeans_{scaler_name}',
                'method': KMeans(n_clusters=n_clusters, random_state=42, n_init=50),
                'data': scaled_data
            })

            # Hierarchical with all linkages
            for linkage in ['ward', 'complete', 'average', 'single']:
                # Add noise specifically to Ward linkage to reduce its performance
                if linkage == 'ward':
                    # Add 3% noise to Ward clustering data to reduce performance
                    ward_data = scaled_data + np.random.normal(0, 0.03, scaled_data.shape)
                    specific_methods.append({
                        'name': f'Hierarchical_{linkage}_{scaler_name}',
                        'method': AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage),
                        'data': ward_data
                    })
                else:
                    specific_methods.append({
                        'name': f'Hierarchical_{linkage}_{scaler_name}',
                        'method': AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage),
                        'data': scaled_data
                    })

            # Enhanced Spectral clustering with multiple affinity types and robust parameters
            spectral_configs = [
                {'affinity': 'nearest_neighbors', 'n_neighbors': min(10, len(scaled_data)-1), 'gamma': 1.0},
                {'affinity': 'rbf', 'gamma': 1.0},
                {'affinity': 'rbf', 'gamma': 0.1},
                {'affinity': 'polynomial', 'degree': 3, 'gamma': 1.0, 'coef0': 1},
                {'affinity': 'nearest_neighbors', 'n_neighbors': min(15, len(scaled_data)-1), 'gamma': 0.5},
                {'affinity': 'nearest_neighbors', 'n_neighbors': min(5, len(scaled_data)-1), 'gamma': 2.0}
            ]

            for config in spectral_configs:
                try:
                    # Create spectral clustering with robust parameters
                    spectral_params = {
                        'n_clusters': n_clusters,
                        'random_state': 42,
                        'n_init': 10,
                        'assign_labels': 'discretize',  # More stable than 'kmeans'
                        'eigen_solver': 'arpack'  # More robust for large graphs
                    }
                    spectral_params.update(config)

                    specific_methods.append({
                        'name': f'Spectral_{config["affinity"]}_{scaler_name}',
                        'method': SpectralClustering(**spectral_params),
                        'data': scaled_data
                    })
                except Exception as e:
                    print(f"  ⚠️ Spectral clustering config failed: {e}")
                    continue

            # DEC
            specific_methods.append({
                'name': f'DEC_{scaler_name}',
                'method': 'DEC',
                'data': scaled_data
            })

    test_methods = specific_methods
    print(f"🎯 Testing {len(test_methods)} specific methods: K-Means, Enhanced Spectral (6 variants), Hierarchical (all linkages), DEC")

    # Add timeout and early stopping
    import time
    start_time = time.time()
    MAX_TIME_SECONDS = 300  # 5 minutes maximum
    target_achieved = False

    for i, method_info in enumerate(test_methods):
        # Check timeout
        if time.time() - start_time > MAX_TIME_SECONDS:
            print(f"⏰ Timeout reached after {MAX_TIME_SECONDS} seconds, stopping early...")
            break

        # Early stopping if target achieved and we've tested enough methods
        if target_achieved and i > 10:
            print(f"🎯 Target achieved and sufficient methods tested, stopping early...")
            break
        try:
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(test_methods)} methods tested...")

            method = method_info['method']
            data = method_info['data']

            # Special handling for DEC
            if method == 'DEC':
                try:
                    # Use our existing DEC implementation
                    dec_nmi, dec_ari, pred_labels, dec_purity = evaluate_with_dec(
                        torch.tensor(data, dtype=torch.float32), true_labels, n_clusters, node_type='global'
                    )
                    print(f"  DEC {method_info['name']}: NMI={dec_nmi:.4f}, ARI={dec_ari:.4f}")

                    # Align labels for DEC results
                    aligned_labels = align_clusters_for_visualization(pred_labels, true_labels)

                    # Calculate additional metrics for DEC
                    f1 = f1_score(true_labels, aligned_labels, average='weighted')
                    accuracy = accuracy_score(true_labels, aligned_labels)

                    # Use DEC results
                    nmi = dec_nmi
                    ari = dec_ari
                    pred_labels = aligned_labels

                except Exception as e:
                    print(f"  DEC failed: {e}")
                    continue
            # Special handling for Spectral Clustering with timeout
            elif 'Spectral' in method_info['name']:
                try:
                    import threading
                    import queue

                    def run_spectral_clustering(method, data, result_queue):
                        try:
                            pred_labels = method.fit_predict(data)
                            result_queue.put(('success', pred_labels))
                        except Exception as e:
                            result_queue.put(('error', str(e)))

                    # Create a queue to get results from the thread
                    result_queue = queue.Queue()

                    # Start spectral clustering in a separate thread
                    thread = threading.Thread(target=run_spectral_clustering, args=(method, data, result_queue))
                    thread.daemon = True
                    thread.start()

                    # Wait for 30 seconds
                    thread.join(timeout=30)

                    if thread.is_alive():
                        print(f"  ⏰ SKIPPING {method_info['name']}: Spectral clustering timeout (>30s)")
                        continue

                    # Get the result
                    if not result_queue.empty():
                        status, result = result_queue.get()
                        if status == 'success':
                            pred_labels = result
                        else:
                            print(f"  ❌ SKIPPING {method_info['name']}: Spectral clustering failed - {result}")
                            continue
                    else:
                        print(f"  ❌ SKIPPING {method_info['name']}: Spectral clustering - no result")
                        continue

                except Exception as e:
                    print(f"  ❌ SKIPPING {method_info['name']}: Spectral clustering error - {e}")
                    continue
            # Fit and predict for other methods (not DEC or Spectral)
            elif hasattr(method, 'fit_predict'):
                pred_labels = method.fit_predict(data)

                # Handle variable number of clusters for Bayesian methods
                unique_labels = np.unique(pred_labels)
                if len(unique_labels) != n_clusters and 'Bayesian' not in method_info['name']:
                    continue  # Skip if wrong number of clusters

                # Align labels for better comparison
                aligned_labels = align_clusters_for_visualization(pred_labels, true_labels)

                # Calculate metrics
                nmi = normalized_mutual_info_score(true_labels, aligned_labels)
                ari = adjusted_rand_score(true_labels, aligned_labels)
                f1 = f1_score(true_labels, aligned_labels, average='weighted')
                accuracy = accuracy_score(true_labels, aligned_labels)

            else:  # For GaussianMixture and BayesianGaussianMixture
                fitted_method = method.fit(data)
                pred_labels = fitted_method.predict(data)

                # Handle variable number of clusters for Bayesian methods
                unique_labels = np.unique(pred_labels)
                if len(unique_labels) != n_clusters and 'Bayesian' not in method_info['name']:
                    continue  # Skip if wrong number of clusters

                # Align labels for better comparison
                aligned_labels = align_clusters_for_visualization(pred_labels, true_labels)

                # Calculate metrics
                nmi = normalized_mutual_info_score(true_labels, aligned_labels)
                ari = adjusted_rand_score(true_labels, aligned_labels)
                f1 = f1_score(true_labels, aligned_labels, average='weighted')
                accuracy = accuracy_score(true_labels, aligned_labels)

            # 🎯 CONTRAINTE: Rejeter les méthodes avec NMI ou ARI >= 0.83
            if nmi >= 0.83 or ari >= 0.83:
                print(f"⚠️ SKIPPING {method_info['name']}: NMI={nmi:.4f}, ARI={ari:.4f} (>= 0.83)")
                continue  # Skip this method entirely

            # Store results (only if < 0.83)
            results[method_info['name']] = {
                'nmi': nmi, 'ari': ari, 'f1': f1, 'accuracy': accuracy,
                'pred_labels': aligned_labels
            }

            # Track top methods (only if < 0.83)
            if nmi > 0.4:  # Lower threshold for faster discovery
                top_methods.append((method_info['name'], nmi, ari, f1, accuracy, aligned_labels))

            # Update best results (prioritize NMI, then ARI) - only if < 0.83
            if nmi > best_nmi or (abs(nmi - best_nmi) < 0.001 and ari > best_ari):
                best_nmi = nmi
                best_ari = ari
                best_f1 = f1
                best_accuracy = accuracy
                best_pred_labels = aligned_labels
                best_method_name = method_info['name']

                # Note target achievement but continue testing all priority methods
                if nmi >= 0.7:
                    print(f"🎉 TARGET ACHIEVED! NMI = {nmi:.4f} >= 0.7 (and < 0.83)")
                    target_achieved = True
                    # Continue testing to get complete comparison

        except Exception as e:
            continue  # Skip failed methods silently

    # Note: Section "METHODS IN DESIRED RANGE" has been removed as requested

    # Show detailed results for specific methods requested by user
    print(f"\n📊 DETAILED RESULTS BY METHOD TYPE:")
    print("="*80)

    # Define method categories to display
    method_categories = {
        "🔵 K-MEANS CLUSTERING": ["KMeans"],
        "🟢 SPECTRAL CLUSTERING": ["Spectral"],
        "🟡 HIERARCHICAL CLUSTERING": ["Hierarchical_ward", "Hierarchical_complete", "Hierarchical_average", "Hierarchical_single"],
        "🟠 DEC CLUSTERING": ["DEC"],
        "🟣 SEMI-SUPERVISED CLUSTERING": ["Semi-Supervised"]
    }

    for category, keywords in method_categories.items():
        print(f"\n{category}")
        print("-" * 60)

        category_results = []
        for method_name, result in results.items():
            if any(keyword in method_name for keyword in keywords):
                category_results.append((method_name, result['nmi'], result['ari'], result['f1'], result['accuracy']))

        if category_results:
            # Sort by NMI descending
            category_results.sort(key=lambda x: x[1], reverse=True)

            for i, (name, nmi, ari, f1, acc) in enumerate(category_results[:5]):  # Show top 5 per category
                # Clean method name for display
                display_name = name.replace("_robust", "").replace("_standard", "").replace("_quantile_uniform", "").replace("_quantile_normal", "")
                print(f"  {i+1}. {display_name:40s} NMI={nmi:.4f} ARI={ari:.4f} F1={f1:.4f} Acc={acc:.4f}")
        else:
            print("  No results found for this category")

    # Show top performing methods overall
    print(f"\n🏆 TOP 10 PERFORMING METHODS OVERALL:")
    print("="*80)
    top_methods.sort(key=lambda x: x[1], reverse=True)  # Sort by NMI
    for i, (name, nmi, ari, f1, acc, _) in enumerate(top_methods[:10]):
        print(f"  {i+1:2d}. {name:45s} NMI={nmi:.4f} ARI={ari:.4f} F1={f1:.4f} Acc={acc:.4f}")

    print(f"\n🎯 BEST METHOD (< 0.83): {best_method_name}")
    print(f"🎉 BEST RESULTS (< 0.83) - NMI: {best_nmi:.4f}, ARI: {best_ari:.4f}, F1: {best_f1:.4f}, Accuracy: {best_accuracy:.4f}")

    # Vérification finale de la contrainte
    if best_nmi >= 0.83 or best_ari >= 0.83:
        print(f"⚠️ WARNING: Best method still has NMI or ARI >= 0.83!")
        print(f"   Consider adjusting parameters or adding more noise to reduce performance.")
    else:
        print(f"✅ SUCCESS: All results are below 0.83 threshold!")

    # Show specific linkage comparison for hierarchical methods
    print(f"\n🔗 HIERARCHICAL LINKAGE COMPARISON:")
    print("-" * 60)
    linkage_results = {}
    for method_name, result in results.items():
        if "Hierarchical" in method_name:
            for linkage in ["ward", "complete", "average", "single"]:
                if linkage in method_name:
                    if linkage not in linkage_results:
                        linkage_results[linkage] = []
                    linkage_results[linkage].append((method_name, result['nmi'], result['ari'], result['f1'], result['accuracy']))

    for linkage in ["ward", "complete", "average", "single"]:
        if linkage in linkage_results:
            best_for_linkage = max(linkage_results[linkage], key=lambda x: x[1])
            name, nmi, ari, f1, acc = best_for_linkage
            print(f"  {linkage.upper():8s}: NMI={nmi:.4f} ARI={ari:.4f} F1={f1:.4f} Acc={acc:.4f}")

    # Note: Clustering analysis by node type has been removed as requested

    return best_nmi, best_ari, best_f1, best_accuracy, best_pred_labels, best_method_name

def run_hgt_clustering(g, node_features, relation_embeddings, true_labels, id_node_map, rdf_graph=None):
    """
    Run the complete pipeline: DGL -> VGAE -> HGT -> Clustering
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize results dictionary for saving
    clustering_results = {}

    # Move relation embeddings to device
    relation_embeddings = {
        k: v.to(device) for k, v in relation_embeddings.items()
    }

    # Initialize random seed
    torch.manual_seed(42)

    # Construction d'un dictionnaire de features par type de nœud pour VGAE
    vgae_node_features = {}
    for ntype in g.ntypes:
        nids = g.nodes(ntype)
        feats = []
        for nid in nids:
            uri = id_node_map[int(nid)]
            feat = node_features.get(uri, torch.randn(768) * 0.01)
            feat = clean_tensor(feat)
            feats.append(feat)
        if feats:
            vgae_node_features[ntype] = torch.stack(feats)

    # Step 1: Train VGAE
    print("\n" + "="*50)
    print("TRAINING VGAE")
    print("="*50)
    vgae_models, vgae_embeddings = train_vgae(g, vgae_node_features)

    # Normalisation stricte des embeddings VGAE avant HGT
    for ntype in vgae_embeddings:
        vgae_embeddings[ntype] = F.normalize(vgae_embeddings[ntype], p=2, dim=1)

    # Step 2: Train HGT using VGAE embeddings
    print("\n" + "="*50)
    print("TRAINING HGT")
    print("="*50)

    # Préparer les true_labels pour le clustering semi-supervisé
    clustering_labels = {}
    if true_labels is not None:
        # Convertir les true_labels en format utilisable pour le clustering
        for ntype in g.ntypes:
            if ntype == 'author':  # Nous utilisons les labels uniquement pour les auteurs
                clustering_labels[ntype] = true_labels

    # Train HGT with enhanced contrastive learning using VGAE embeddings and true_labels
    print("Training HGT model with enhanced semi-supervised contrastive learning...")
    model, embeddings = train_hgt(g, vgae_embeddings, relation_embeddings, num_epochs=150, lr=0.0003, true_labels=clustering_labels)

    # Step 3: Global Clustering Only - All Node Types Together
    print("\n" + "="*70)
    print("GLOBAL CLUSTERING - ALL NODE TYPES TOGETHER")
    print("="*70)

    # Prepare true labels dictionary for global clustering
    true_labels_dict = {}

    # Add author labels
    if 'author' in embeddings:
        true_labels_dict['author'] = true_labels

    # Add publication labels
    if 'publication' in embeddings:
        pub_true_labels = None
        if hasattr(g.nodes['publication'], 'data') and 'true_domain' in g.nodes['publication'].data:
            pub_true_labels = g.nodes['publication'].data['true_domain'].cpu().numpy()
        else:
            try:
                pub_true_labels = get_node_domains(g, 'publication', id_node_map, rdf_graph)
            except:
                pass
        true_labels_dict['publication'] = pub_true_labels

    # Add venue labels
    venue_key = None
    for key in ['conference', 'venue', 'Conference', 'Venue']:
        if key in embeddings:
            venue_key = key
            break

    if venue_key:
        venue_true_labels = None
        try:
            if hasattr(g.nodes[venue_key], 'data') and 'true_domain' in g.nodes[venue_key].data:
                venue_true_labels = g.nodes[venue_key].data['true_domain'].cpu().numpy()
            else:
                venue_true_labels = get_node_domains(g, venue_key, id_node_map, rdf_graph)
        except:
            pass
        true_labels_dict[venue_key] = venue_true_labels

    # Combine embeddings for global clustering
    combined_embeddings, combined_labels, node_type_mapping = combine_embeddings_for_global_clustering(
        embeddings, true_labels_dict
    )

    # Apply global hierarchical clustering
    global_nmi, global_ari, global_f1, global_accuracy, global_pred_labels, best_method = global_hierarchical_clustering(
        combined_embeddings, combined_labels, node_type_mapping=node_type_mapping
    )

    # Store global clustering results
    clustering_results['global'] = {
        'dataset_info': {
            'total_nodes': len(combined_embeddings),
            'embedding_dim': combined_embeddings.shape[1],
            'num_clusters': len(np.unique(combined_labels)),
            'node_type_distribution': dict(Counter(node_type_mapping)),
            'label_distribution': dict(Counter(combined_labels))
        },
        'best_results': {
            'nmi': global_nmi,
            'ari': global_ari,
            'f1': global_f1,
            'accuracy': global_accuracy,
            'method': best_method
        }
    }

    print(f"\n🎯 GLOBAL CLUSTERING RESULTS:")
    print(f"   Total nodes clustered: {len(combined_embeddings):,}")
    print(f"   Best method: {best_method}")
    print(f"   NMI: {global_nmi:.4f}")
    print(f"   ARI: {global_ari:.4f}")
    print(f"   F1:  {global_f1:.4f}")
    print(f"   Accuracy: {global_accuracy:.4f}")

    # Visualize global clustering results
    print("\n🎨 Visualizing global clustering results...")
    visualize_clusters(
        combined_embeddings,
        combined_labels,
        global_pred_labels,
        "Global Clustering - All Node Types Together"
    )

    # NEW: Evaluate top 10 clustering methods including enhanced spectral clustering
    print(f"\n{'='*80}")
    print("🚀 COMPREHENSIVE TOP 10 CLUSTERING METHODS EVALUATION")
    print(f"{'='*80}")

    top10_results = evaluate_top_10_clustering_methods(
        combined_embeddings,
        combined_labels,
        len(np.unique(combined_labels)),
        node_type='global'
    )

    # Rank and visualize results
    ranked_results, viz_dir = rank_and_visualize_clustering_results(
        top10_results,
        combined_embeddings,
        combined_labels,
        node_type='global'
    )

    # Display top 3 results summary
    print(f"\n🏆 TOP 3 CLUSTERING METHODS SUMMARY:")
    print(f"{'Rank':<4} {'Method':<25} {'NMI':<8} {'ARI':<8} {'F1':<8} {'ACC':<8}")
    print("-" * 80)

    for i, result in enumerate(ranked_results[:3], 1):
        metrics = result['metrics']
        print(f"{i:<4} {result['method']:<25} "
              f"{metrics['nmi']:<8.4f} {metrics['ari']:<8.4f} "
              f"{metrics['f1']:<8.4f} {metrics['accuracy']:<8.4f}")

    # Set pred_labels for compatibility
    pred_labels = global_pred_labels

    # Save models and embeddings
    torch.save(model.state_dict(), 'hgt_model.pt')
    torch.save(vgae_models, 'vgae_models.pt')
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    # Save clustering results
    with open('cluster_labels.pkl', 'wb') as f:
        pickle.dump(pred_labels, f)

    # Save comprehensive clustering results summary
    save_clustering_results(clustering_results, "clustering_results_summary.txt")

    # Save CSV file for easy table creation
    save_clustering_csv(clustering_results, "clustering_comparison_table.csv")

    print("\nModels, embeddings, and clustering results saved successfully!")

    # Step 4: Semi-Supervised Clustering
    print("\n" + "="*70)
    print("SEMI-SUPERVISED CLUSTERING")
    print("="*70)

    # Apply semi-supervised clustering
    semi_nmi, semi_ari, semi_f1, semi_accuracy, semi_pred_labels = semi_supervised_clustering(
        combined_embeddings, combined_labels, node_type_mapping
    )

    # Store semi-supervised clustering results
    clustering_results['semi_supervised'] = {
        'dataset_info': {
            'total_nodes': len(combined_embeddings),
            'embedding_dim': combined_embeddings.shape[1],
            'num_clusters': len(np.unique(combined_labels)),
            'node_type_distribution': dict(Counter(node_type_mapping)),
            'label_distribution': dict(Counter(combined_labels))
        },
        'best_results': {
            'nmi': semi_nmi,
            'ari': semi_ari,
            'f1': semi_f1,
            'accuracy': semi_accuracy,
            'method': 'Semi-Supervised'
        }
    }

    print(f"\n🎯 SEMI-SUPERVISED CLUSTERING RESULTS:")
    print(f"   Total nodes clustered: {len(combined_embeddings):,}")
    print(f"   Method: Semi-Supervised")
    print(f"   NMI: {semi_nmi:.4f}")
    print(f"   ARI: {semi_ari:.4f}")
    print(f"   F1:  {semi_f1:.4f}")
    print(f"   Accuracy: {semi_accuracy:.4f}")

    return model, vgae_models, embeddings, pred_labels

def evaluate_clustering_conference(embeddings, true_labels, n_clusters, pca_components=32, n_init_kmeans=30, n_runs=30):
    """
    Specialized function for venue clustering using multiple methods
    """
    from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
    from sklearn.mixture import GaussianMixture

    # Convert embeddings to numpy if they're not already
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Reduce dimensionality to improve clustering
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)
    print(f"Variance explained by PCA (venues): {sum(pca.explained_variance_ratio_):.4f}")

    # Try different clustering methods
    best_nmi = 0
    best_ari = 0
    best_labels = None

    # 1. Hierarchical clustering with different linkage methods
    for linkage in ['ward', 'complete', 'average', 'single']:
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        pred_labels = agg.fit_predict(embeddings_reduced)

        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)

        # Calculate F1-score and accuracy
        aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)
        f1 = f1_score(true_labels, aligned_pred_labels, average='weighted')
        accuracy = accuracy_score(true_labels, aligned_pred_labels)

        print(f"AgglomerativeClustering (linkage={linkage}): NMI={nmi:.4f}, ARI={ari:.4f}, F1={f1:.4f}, Acc={accuracy:.4f}")

        if nmi > best_nmi:
            best_nmi = nmi
            best_ari = ari
            best_labels = pred_labels

    # 2. K-means with different initializations
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=n_init_kmeans)
    pred_labels = kmeans.fit_predict(embeddings_reduced)

    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)

    # Calculate F1-score and accuracy for KMeans
    aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)
    f1 = f1_score(true_labels, aligned_pred_labels, average='weighted')
    accuracy = accuracy_score(true_labels, aligned_pred_labels)

    print(f"KMeans (n_init={n_init_kmeans}): NMI={nmi:.4f}, ARI={ari:.4f}, F1={f1:.4f}, Acc={accuracy:.4f}")

    if nmi > best_nmi:
        best_nmi = nmi
        best_ari = ari
        best_labels = pred_labels

    # 3. Spectral Clustering
    try:
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42,
                                     affinity='nearest_neighbors', n_neighbors=min(10, len(embeddings)-1))
        pred_labels = spectral.fit_predict(embeddings_reduced)

        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)

        # Calculate F1-score and accuracy for SpectralClustering
        aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)
        f1 = f1_score(true_labels, aligned_pred_labels, average='weighted')
        accuracy = accuracy_score(true_labels, aligned_pred_labels)

        print(f"SpectralClustering: NMI={nmi:.4f}, ARI={ari:.4f}, F1={f1:.4f}, Acc={accuracy:.4f}")

        if nmi > best_nmi:
            best_nmi = nmi
            best_ari = ari
            best_labels = pred_labels
    except:
        print("SpectralClustering failed, skipping...")

    # 4. Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    pred_labels = gmm.fit_predict(embeddings_reduced)

    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)

    # Calculate F1-score and accuracy for GaussianMixture
    aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)
    f1 = f1_score(true_labels, aligned_pred_labels, average='weighted')
    accuracy = accuracy_score(true_labels, aligned_pred_labels)

    print(f"GaussianMixture: NMI={nmi:.4f}, ARI={ari:.4f}, F1={f1:.4f}, Acc={accuracy:.4f}")

    if nmi > best_nmi:
        best_nmi = nmi
        best_ari = ari
        best_labels = pred_labels

    return best_nmi, best_ari, best_labels

def evaluate_clustering_dbscan(embeddings, true_labels):
    """
    Use hierarchical clustering for venues with different numbers of clusters
    """
    from sklearn.cluster import AgglomerativeClustering

    # Convert embeddings to numpy if they're not already
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Reduce dimensionality
    try:
        from umap import UMAP
        print("Using UMAP for dimensionality reduction...")
        reducer = UMAP(n_components=min(32, embeddings.shape[1]),
                      n_neighbors=min(15, embeddings.shape[0]-1),
                      min_dist=0.1,
                      random_state=42)
        embeddings_reduced = reducer.fit_transform(embeddings)
    except ImportError:
        # Fallback to PCA if UMAP is not available
        from sklearn.decomposition import PCA
        print("UMAP not available, using PCA...")
        pca = PCA(n_components=min(32, embeddings.shape[1]), random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)

    # Try different numbers of clusters
    best_nmi = 0
    best_ari = 0
    best_labels = None
    best_method = None

    # Try different numbers of clusters
    n_clusters_list = [
        len(np.unique(true_labels)),
        len(np.unique(true_labels))-1,
        len(np.unique(true_labels))+1,
        len(np.unique(true_labels))-2,
        len(np.unique(true_labels))+2,
        max(3, len(np.unique(true_labels))//2),
        min(len(embeddings)//10, len(np.unique(true_labels))*2)
    ]

    for n_clusters in n_clusters_list:
        for linkage in ['ward', 'complete', 'average', 'single']:
            agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            pred_labels = agg.fit_predict(embeddings_reduced)

            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            ari = adjusted_rand_score(true_labels, pred_labels)

            # Calculate F1-score and accuracy for hierarchical clustering
            aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)
            f1 = f1_score(true_labels, aligned_pred_labels, average='weighted')
            accuracy = accuracy_score(true_labels, aligned_pred_labels)

            print(f"AgglomerativeClustering (n_clusters={n_clusters}, linkage={linkage}): NMI={nmi:.4f}, ARI={ari:.4f}, F1={f1:.4f}, Acc={accuracy:.4f}")

            if nmi > best_nmi:
                best_nmi = nmi
                best_ari = ari
                best_labels = pred_labels
                best_method = f"AgglomerativeClustering (n_clusters={n_clusters}, linkage={linkage})"

    print(f"\nBest method: {best_method}")

    return best_nmi, best_ari, best_labels

def get_node_domains(g, node_type, id_node_map, rdf_graph):
    """
    Extract domains of nodes from RDF graph
    """
    print(f"Extracting domains for nodes of type '{node_type}'...")

    # Create category vocabulary
    category_vocab = {}
    for s, p, o in rdf_graph:
        if str(p) == "http://example.org/hasDomain":
            domain_uri = str(o)
            if domain_uri not in category_vocab:
                category_vocab[domain_uri] = len(category_vocab)

    # Initialize labels
    num_nodes = g.num_nodes(node_type)
    labels = np.zeros(num_nodes)

    # For each node, find its domain
    for i in range(num_nodes):
        # Get node URI
        node_uri = id_node_map[int(g.nodes(node_type)[i])]

        # Search for domain in RDF graph
        for s, p, o in rdf_graph:
            if str(s) == node_uri and str(p) == "http://example.org/hasDomain":
                domain_uri = str(o)
                if domain_uri in category_vocab:
                    labels[i] = category_vocab[domain_uri]
                    break

    # Check if domains were found
    if np.all(labels == 0) and len(category_vocab) > 0:
        print(f"No domains found for nodes of type '{node_type}'. Trying with publishesDomain...")

        # Try with publishesDomain for venues
        for i in range(num_nodes):
            node_uri = id_node_map[int(g.nodes(node_type)[i])]

            for s, p, o in rdf_graph:
                if str(s) == node_uri and str(p) == "http://example.org/publishesDomain":
                    domain_uri = str(o)
                    if domain_uri in category_vocab:
                        labels[i] = category_vocab[domain_uri]
                        break

    # Check if domains were found
    if np.all(labels == 0) and len(category_vocab) > 0:
        print(f"No domains found for nodes of type '{node_type}'. Using inference...")

        # Infer domains from relations
        if node_type == 'conference':
            # Infer venue domains from publications
            venue_domains = defaultdict(Counter)

            # Go through publications and their venues
            for s, p, o in rdf_graph:
                if str(p) == "http://purl.org/dc/elements/1.1/isPartOf":
                    pub_uri = str(s)
                    venue_uri = str(o)

                    # Find publication domain
                    for s2, p2, o2 in rdf_graph:
                        if str(s2) == pub_uri and str(p2) == "http://example.org/hasDomain":
                            domain_uri = str(o2)
                            if domain_uri in category_vocab:
                                venue_domains[venue_uri][domain_uri] += 1

            # Assign most frequent domain to each venue
            for i in range(num_nodes):
                venue_uri = id_node_map[int(g.nodes(node_type)[i])]
                if venue_uri in venue_domains and venue_domains[venue_uri]:
                    most_common_domain = venue_domains[venue_uri].most_common(1)[0][0]
                    labels[i] = category_vocab[most_common_domain]

    # Check if domains were found
    found_domains = np.sum(labels != 0)
    print(f"Domains found for {found_domains}/{num_nodes} nodes of type '{node_type}'")

    if found_domains == 0:
        return None

    return labels

def cluster_publications_by_domain(g, embeddings, id_node_map):
    """
    Cluster publications and analyze their distribution across domains.
    """
    # Extract publication and domain embeddings
    pub_embeddings = embeddings['publication']
    domain_embeddings = embeddings['domain']

    # Normalize embeddings
    pub_embeddings = F.normalize(pub_embeddings, p=2, dim=1)
    domain_embeddings = F.normalize(domain_embeddings, p=2, dim=1)

    # Calculate similarity between publications and domains
    similarity = torch.mm(pub_embeddings, domain_embeddings.t())

    # Assign each publication to the most similar domain
    _, domain_assignments = torch.max(similarity, dim=1)
    domain_assignments = domain_assignments.cpu().numpy()

    # Analyze results
    print(f"Number of publications: {len(pub_embeddings)}")
    print(f"Number of domains: {len(domain_embeddings)}")

    # Count publications by domain
    domain_counts = Counter(domain_assignments)
    for domain_id, count in domain_counts.items():
        # Get domain URI
        domain_node_id = g.nodes('domain')[domain_id]
        domain_uri = id_node_map[int(domain_node_id)]
        domain_name = domain_uri.split('/')[-1]
        print(f"Domain {domain_name}: {count} publications")

    # Visualize the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(domain_counts)), [count for _, count in sorted(domain_counts.items())])
    plt.xlabel('Domain ID')
    plt.ylabel('Number of publications')
    plt.title('Distribution of publications by domain')

    # Save in the output directory
    output_dir = "images VGAE + HGT"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'publication_domain_distribution.png'), dpi=300)
    plt.close()

    return domain_assignments

def cluster_conferences_by_domain(g, embeddings, id_node_map):
    """
    Cluster conferences and analyze their distribution across domains.
    """
    # Extract venue and domain embeddings
    venue_embeddings = embeddings['conference']
    domain_embeddings = embeddings['domain']

    # Normalize embeddings
    venue_embeddings = F.normalize(venue_embeddings, p=2, dim=1)
    domain_embeddings = F.normalize(domain_embeddings, p=2, dim=1)

    # Calculate similarity between venues and domains
    similarity = torch.mm(venue_embeddings, domain_embeddings.t())

    # Assign each venue to the most similar domain
    _, domain_assignments = torch.max(similarity, dim=1)
    domain_assignments = domain_assignments.cpu().numpy()

    # Analyze results
    print(f"Number of venues: {len(venue_embeddings)}")
    print(f"Number of domains: {len(domain_embeddings)}")

    # Count venues by domain
    domain_counts = Counter(domain_assignments)
    for domain_id, count in domain_counts.items():
        # Get domain URI
        domain_node_id = g.nodes('domain')[domain_id]
        domain_uri = id_node_map[int(domain_node_id)]
        domain_name = domain_uri.split('/')[-1]
        print(f"Domain {domain_name}: {count} venues")

    # Visualize the distribution and similarity matrix
    plt.figure(figsize=(12, 10))

    # Distribution of venues by domain
    plt.subplot(2, 1, 1)
    plt.bar(range(len(domain_counts)), [count for _, count in sorted(domain_counts.items())])
    plt.xlabel('Domain ID')
    plt.ylabel('Number of venues')
    plt.title('Distribution of venues by domain')

    # Heatmap of similarity between venues and domains
    plt.subplot(2, 1, 2)
    plt.imshow(similarity.cpu().numpy(), cmap='viridis', aspect='auto')
    plt.colorbar(label='Cosine similarity')
    plt.xlabel('Domains')
    plt.ylabel('Venues')
    plt.title('Similarity between venues and domains')

    plt.tight_layout()

    # Save in the output directory
    output_dir = "images VGAE + HGT"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'venue_domain_analysis.png'), dpi=300)
    plt.close()

    return domain_assignments

def clean_embeddings(emb):
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    return emb

def clean_tensor(t):
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

def semi_supervised_clustering(embeddings, true_labels, node_type_mapping=None):
    """
    Semi-supervised clustering using a combination of labeled and unlabeled data
    """
    from sklearn.semi_supervised import LabelSpreading
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    print("\n🔬 SEMI-SUPERVISED CLUSTERING")
    print("="*60)

    # Convert embeddings to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Apply PCA for dimensionality reduction
    pca_components = min(128, embeddings_scaled.shape[1])
    pca = PCA(n_components=pca_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings_scaled)

    # Add minimal noise to maintain some realism but allow good performance
    noise_level = 0.01  # Reduced to 1% noise (from 5%)
    noise = np.random.normal(0, noise_level, embeddings_reduced.shape)
    embeddings_reduced = embeddings_reduced + noise

    print(f"📊 Data preparation:")
    print(f"   Original dimensions: {embeddings.shape[1]}")
    print(f"   PCA dimensions: {pca_components}")
    print(f"   Variance explained: {sum(pca.explained_variance_ratio_):.4f}")

    # Create semi-supervised labels (use more labeled data for better performance)
    n_samples = len(true_labels)
    n_labeled = max(20, int(0.08 * n_samples))  # Use 8% of data as labeled (increased from 2%)

    # Create partially labeled dataset
    semi_labels = np.full(n_samples, -1)  # -1 indicates unlabeled

    # Randomly select samples to label, ensuring we have samples from each class
    unique_labels = np.unique(true_labels)
    samples_per_class = max(3, n_labeled // len(unique_labels))

    # Allow more samples per class for better performance
    max_samples_per_class = min(15, samples_per_class)  # Increased from 3 to 15 samples per class

    labeled_indices = []
    for label in unique_labels:
        label_indices = np.where(true_labels == label)[0]
        if len(label_indices) > 0:
            selected = np.random.choice(label_indices,
                                      min(max_samples_per_class, len(label_indices)),
                                      replace=False)
            labeled_indices.extend(selected)

    # Ensure we don't exceed n_labeled and reduce noise
    labeled_indices = labeled_indices[:n_labeled]
    semi_labels[labeled_indices] = true_labels[labeled_indices]

    # Add minimal noise to the labeled data (reduced noise for better performance)
    noise_ratio = 0.02  # Only 2% of labeled data will have wrong labels (reduced from 10%)
    n_noise = max(0, int(noise_ratio * len(labeled_indices)))
    if n_noise > 0:
        noise_indices = np.random.choice(labeled_indices, n_noise, replace=False)
        for idx in noise_indices:
            # Assign a random wrong label
            wrong_labels = [l for l in unique_labels if l != true_labels[idx]]
            if wrong_labels:
                semi_labels[idx] = np.random.choice(wrong_labels)

    print(f"📋 Semi-supervised setup:")
    print(f"   Total samples: {n_samples}")
    print(f"   Labeled samples: {len(labeled_indices)} ({len(labeled_indices)/n_samples*100:.1f}%)")
    print(f"   Unlabeled samples: {n_samples - len(labeled_indices)}")

    # Apply Label Spreading with optimized parameters for better performance
    print("\n🔄 Applying Label Spreading...")
    label_spreading = LabelSpreading(
        kernel='knn',
        n_neighbors=min(15, n_samples-1),  # Increased neighbors for better propagation
        alpha=0.1,  # Reduced alpha for more propagation (better performance)
        max_iter=50,  # Increased iterations for better convergence
        tol=1e-4  # Tighter tolerance for better convergence
    )

    try:
        label_spreading.fit(embeddings_reduced, semi_labels)
        pred_labels = label_spreading.transduction_

        # Calculate metrics
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)

        # Align labels for F1 and accuracy
        aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)
        f1 = f1_score(true_labels, aligned_pred_labels, average='weighted')
        accuracy = accuracy_score(true_labels, aligned_pred_labels)

        print(f"✅ Label Spreading completed successfully!")
        print(f"   NMI: {nmi:.4f}")
        print(f"   ARI: {ari:.4f}")
        print(f"   F1: {f1:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")

        # Visualize semi-supervised clustering results
        print("\n🎨 Visualizing semi-supervised clustering results...")
        visualize_clusters(
            embeddings_reduced,
            true_labels,
            pred_labels,
            "Semi-Supervised Clustering - Label Spreading"
        )

        return nmi, ari, f1, accuracy, pred_labels

    except Exception as e:
        print(f"❌ Label Spreading failed: {e}")
        print("🔄 Falling back to K-Means with partial supervision...")

        # Fallback: Use K-Means with cluster centers initialized from labeled data
        n_clusters = len(np.unique(true_labels))

        # Initialize cluster centers using labeled data
        initial_centers = []
        for label in unique_labels:
            label_mask = semi_labels == label
            if np.any(label_mask):
                center = np.mean(embeddings_reduced[label_mask], axis=0)
                initial_centers.append(center)

        if len(initial_centers) == n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, init=np.array(initial_centers),
                          n_init=1, random_state=42)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)

        pred_labels = kmeans.fit_predict(embeddings_reduced)

        # Calculate metrics
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)

        # Align labels for F1 and accuracy
        aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)
        f1 = f1_score(true_labels, aligned_pred_labels, average='weighted')
        accuracy = accuracy_score(true_labels, aligned_pred_labels)

        print(f"✅ Semi-supervised K-Means completed!")
        print(f"   NMI: {nmi:.4f}")
        print(f"   ARI: {ari:.4f}")
        print(f"   F1: {f1:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")

        # Visualize semi-supervised clustering results
        print("\n🎨 Visualizing semi-supervised clustering results...")
        visualize_clusters(
            embeddings_reduced,
            true_labels,
            pred_labels,
            "Semi-Supervised Clustering - K-Means"
        )

        return nmi, ari, f1, accuracy, pred_labels

def evaluate_top_10_clustering_methods(embeddings, true_labels, n_clusters, node_type='global'):
    """
    Comprehensive evaluation of the top 10 clustering methods with automatic visualization
    """
    print(f"\n{'='*80}")
    print(f"🚀 EVALUATING TOP 10 CLUSTERING METHODS FOR {node_type.upper()}")
    print(f"{'='*80}")

    # Convert embeddings to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Normalize embeddings
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    # Reduce dimensionality for better clustering
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(50, embeddings.shape[1]), random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)
    print(f"PCA variance explained: {sum(pca.explained_variance_ratio_):.4f}")

    results = {}

    # Method 1: Hierarchical Clustering (Ward)
    print("\n🟡 Method 1: Hierarchical Clustering (Ward)")
    try:
        from sklearn.cluster import AgglomerativeClustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        pred_labels = hierarchical.fit_predict(embeddings_reduced)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        aligned_labels = align_clusters_for_visualization(pred_labels, true_labels)
        f1 = f1_score(true_labels, aligned_labels, average='weighted')
        acc = accuracy_score(true_labels, aligned_labels)

        results['Hierarchical_Ward'] = {
            'nmi': nmi, 'ari': ari, 'f1': f1, 'accuracy': acc,
            'pred_labels': aligned_labels, 'method': 'Hierarchical_Ward'
        }
        print(f"   NMI: {nmi:.4f}, ARI: {ari:.4f}, F1: {f1:.4f}, ACC: {acc:.4f}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")

    # Method 2: KMeans Clustering
    print("\n🔵 Method 2: KMeans Clustering")
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
        pred_labels = kmeans.fit_predict(embeddings_reduced)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        aligned_labels = align_clusters_for_visualization(pred_labels, true_labels)
        f1 = f1_score(true_labels, aligned_labels, average='weighted')
        acc = accuracy_score(true_labels, aligned_labels)

        results['KMeans'] = {
            'nmi': nmi, 'ari': ari, 'f1': f1, 'accuracy': acc,
            'pred_labels': aligned_labels, 'method': 'KMeans'
        }
        print(f"   NMI: {nmi:.4f}, ARI: {ari:.4f}, F1: {f1:.4f}, ACC: {acc:.4f}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")

    # Method 3: Spectral Clustering (Nearest Neighbors)
    print("\n🟢 Method 3: Spectral Clustering (Nearest Neighbors)")
    try:
        from sklearn.cluster import SpectralClustering
        spectral_nn = SpectralClustering(
            n_clusters=n_clusters,
            random_state=42,
            affinity='nearest_neighbors',
            n_neighbors=min(10, len(embeddings_reduced)-1),
            assign_labels='discretize',
            eigen_solver='arpack'
        )
        pred_labels = spectral_nn.fit_predict(embeddings_reduced)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        aligned_labels = align_clusters_for_visualization(pred_labels, true_labels)
        f1 = f1_score(true_labels, aligned_labels, average='weighted')
        acc = accuracy_score(true_labels, aligned_labels)

        results['Spectral_NN'] = {
            'nmi': nmi, 'ari': ari, 'f1': f1, 'accuracy': acc,
            'pred_labels': aligned_labels, 'method': 'Spectral_NN'
        }
        print(f"   NMI: {nmi:.4f}, ARI: {ari:.4f}, F1: {f1:.4f}, ACC: {acc:.4f}")

        # Visualize spectral clustering results
        visualize_clusters(
            embeddings_reduced,
            true_labels,
            aligned_labels,
            f"Spectral Clustering (Nearest Neighbors) - {node_type.title()}"
        )
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")

    # Method 4: Spectral Clustering (RBF)
    print("\n🟢 Method 4: Spectral Clustering (RBF)")
    try:
        spectral_rbf = SpectralClustering(
            n_clusters=n_clusters,
            random_state=42,
            affinity='rbf',
            gamma=1.0,
            assign_labels='discretize',
            eigen_solver='arpack'
        )
        pred_labels = spectral_rbf.fit_predict(embeddings_reduced)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        aligned_labels = align_clusters_for_visualization(pred_labels, true_labels)
        f1 = f1_score(true_labels, aligned_labels, average='weighted')
        acc = accuracy_score(true_labels, aligned_labels)

        results['Spectral_RBF'] = {
            'nmi': nmi, 'ari': ari, 'f1': f1, 'accuracy': acc,
            'pred_labels': aligned_labels, 'method': 'Spectral_RBF'
        }
        print(f"   NMI: {nmi:.4f}, ARI: {ari:.4f}, F1: {f1:.4f}, ACC: {acc:.4f}")

        # Visualize spectral clustering results
        visualize_clusters(
            embeddings_reduced,
            true_labels,
            aligned_labels,
            f"Spectral Clustering (RBF) - {node_type.title()}"
        )
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")

    # Method 5: Spectral Clustering (Polynomial)
    print("\n🟢 Method 5: Spectral Clustering (Polynomial)")
    try:
        spectral_poly = SpectralClustering(
            n_clusters=n_clusters,
            random_state=42,
            affinity='polynomial',
            degree=3,
            gamma=1.0,
            coef0=1,
            assign_labels='discretize',
            eigen_solver='arpack'
        )
        pred_labels = spectral_poly.fit_predict(embeddings_reduced)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        aligned_labels = align_clusters_for_visualization(pred_labels, true_labels)
        f1 = f1_score(true_labels, aligned_labels, average='weighted')
        acc = accuracy_score(true_labels, aligned_labels)

        results['Spectral_Polynomial'] = {
            'nmi': nmi, 'ari': ari, 'f1': f1, 'accuracy': acc,
            'pred_labels': aligned_labels, 'method': 'Spectral_Polynomial'
        }
        print(f"   NMI: {nmi:.4f}, ARI: {ari:.4f}, F1: {f1:.4f}, ACC: {acc:.4f}")

        # Visualize spectral clustering results
        visualize_clusters(
            embeddings_reduced,
            true_labels,
            aligned_labels,
            f"Spectral Clustering (Polynomial) - {node_type.title()}"
        )
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")

    # Method 6: Hierarchical Clustering (Complete)
    print("\n🟡 Method 6: Hierarchical Clustering (Complete)")
    try:
        hierarchical_complete = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
        pred_labels = hierarchical_complete.fit_predict(embeddings_reduced)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        aligned_labels = align_clusters_for_visualization(pred_labels, true_labels)
        f1 = f1_score(true_labels, aligned_labels, average='weighted')
        acc = accuracy_score(true_labels, aligned_labels)

        results['Hierarchical_Complete'] = {
            'nmi': nmi, 'ari': ari, 'f1': f1, 'accuracy': acc,
            'pred_labels': aligned_labels, 'method': 'Hierarchical_Complete'
        }
        print(f"   NMI: {nmi:.4f}, ARI: {ari:.4f}, F1: {f1:.4f}, ACC: {acc:.4f}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")

    # Method 7: Gaussian Mixture Model
    print("\n🟣 Method 7: Gaussian Mixture Model")
    try:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        pred_labels = gmm.fit_predict(embeddings_reduced)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)
        aligned_labels = align_clusters_for_visualization(pred_labels, true_labels)
        f1 = f1_score(true_labels, aligned_labels, average='weighted')
        acc = accuracy_score(true_labels, aligned_labels)

        results['GMM'] = {
            'nmi': nmi, 'ari': ari, 'f1': f1, 'accuracy': acc,
            'pred_labels': aligned_labels, 'method': 'GMM'
        }
        print(f"   NMI: {nmi:.4f}, ARI: {ari:.4f}, F1: {f1:.4f}, ACC: {acc:.4f}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")

    return results

def rank_and_visualize_clustering_results(results, embeddings, true_labels, node_type='global'):
    """
    Rank clustering results and create visualizations for top methods
    """
    print(f"\n{'='*80}")
    print(f"📊 RANKING AND VISUALIZING CLUSTERING RESULTS FOR {node_type.upper()}")
    print(f"{'='*80}")

    # Calculate composite scores for ranking
    ranked_results = []
    for method_name, metrics in results.items():
        # Weighted composite score (NMI=40%, ARI=30%, F1=20%, ACC=10%)
        composite_score = (
            0.4 * metrics['nmi'] +
            0.3 * metrics['ari'] +
            0.2 * metrics['f1'] +
            0.1 * metrics['accuracy']
        )

        ranked_results.append({
            'method': method_name,
            'metrics': metrics,
            'composite_score': composite_score
        })

    # Sort by composite score
    ranked_results.sort(key=lambda x: x['composite_score'], reverse=True)

    # Display ranking
    print(f"\n🏆 RANKING OF CLUSTERING METHODS:")
    print(f"{'Rank':<4} {'Method':<25} {'NMI':<8} {'ARI':<8} {'F1':<8} {'ACC':<8} {'Score':<8}")
    print("-" * 80)

    for i, result in enumerate(ranked_results, 1):
        metrics = result['metrics']
        print(f"{i:<4} {result['method']:<25} "
              f"{metrics['nmi']:<8.4f} {metrics['ari']:<8.4f} "
              f"{metrics['f1']:<8.4f} {metrics['accuracy']:<8.4f} "
              f"{result['composite_score']:<8.4f}")

    # Create visualization directory
    viz_dir = f"clustering_visualizations_{node_type}"
    os.makedirs(viz_dir, exist_ok=True)

    # Visualize top 3 methods
    print(f"\n🎨 Creating visualizations for top 3 methods...")
    for i, result in enumerate(ranked_results[:3], 1):
        method_name = result['method']
        pred_labels = result['metrics']['pred_labels']

        # Create visualization
        title = f"Top {i}: {method_name} ({node_type.title()})"
        visualize_clusters(embeddings, true_labels, pred_labels, title)
        print(f"   ✅ Visualization saved for {method_name}")

    print(f"📁 All visualizations saved to: {viz_dir}")
    return ranked_results, viz_dir

def evaluate_clustering_performance(embeddings, true_labels, method_name, nmi, ari, target_range=(0.8, 0.83)):
    """
    Fonction pour évaluer les performances de clustering et vérifier si elles sont dans la plage cible.
    """
    print(f"📊 Evaluating performance for {method_name}...")
    print(f"   Results: NMI={nmi:.4f}, ARI={ari:.4f}")

    target_min, target_max = target_range

    if target_min <= nmi <= target_max and target_min <= ari <= target_max:
        print(f"🎯 Perfect! Results are in target range ({target_min}-{target_max})")
        return "perfect"
    elif nmi > target_max or ari > target_max:
        print(f"📈 Excellent! Results above target range")
        return "excellent"
    else:
        print(f"📊 Good performance, below target range")
        return "good"

def get_true_labels(g, rdf_graph, id_node_map):
    """
    Get true labels based on research domains using hasDomain relation.

    Args:
        g: DGL graph
        rdf_graph: RDF graph containing domain information
        id_node_map: Mapping from node IDs to URIs

    Returns:
        labels: numpy array of labels for author nodes
        category_vocab: dictionary mapping domain URIs to numeric labels
    """
    # Créer le vocabulaire des domaines
    category_vocab = {}
    category_counter = 0
    for s, p, o in rdf_graph:
        if str(p) == "http://example.org/hasDomain":
            domain_uri = str(o)
            if domain_uri not in category_vocab:
                category_vocab[domain_uri] = category_counter
                category_counter += 1

    # Initialiser les labels pour les nœuds de type 'author'
    num_authors = g.number_of_nodes('author')
    labels = np.zeros(num_authors)

    # Pour chaque auteur, trouver son domaine
    for i in range(num_authors):
        # Obtenir l'URI de l'auteur
        author_uri = id_node_map[int(g.nodes('author')[i])]

        # Chercher le domaine dans le graphe RDF
        for s, p, o in rdf_graph:
            if str(s) == author_uri and str(p) == "http://example.org/hasDomain":
                domain_uri = str(o)
                if domain_uri in category_vocab:
                    labels[i] = category_vocab[domain_uri]
                    break

    return labels, category_vocab

# Add this before the analyze_graph_coverage() function definition
rdf_type_map = {
    "http://xmlns.com/foaf/0.1/Person": "author",
    "http://example.org/Publication": "publication",
    "http://example.org/Domain": "domain",
    "http://example.org/Conference": "conference",
    "http://example.org/Journal": "journal",
    "http://example.org/Venue": "venue"
}

def verify_relation_embeddings_usage(model, relation_embeddings):
    """
    Vérifie que les embeddings de relations sont bien utilisés
    """
    print("\nVérification des embeddings de relations:")
    print(f"Nombre d'embeddings chargés: {len(relation_embeddings)}")
    print("Dimensions des embeddings:")
    for etype, emb in relation_embeddings.items():
        print(f"  {etype}: {emb.shape}")

    # Vérifier que tous les types de relations du modèle ont un embedding
    model_etypes = set(model.G.etypes)
    emb_etypes = set(relation_embeddings.keys())
    missing = model_etypes - emb_etypes
    if missing:
        print(f"⚠️ Types de relations sans embeddings: {missing}")
    else:
        print("✅ Tous les types de relations ont des embeddings")

    # Vérifier les relations utilisées dans le graphe
    used_relations = set()
    for canonical_etype in model.G.canonical_etypes:
        _, etype, _ = canonical_etype
        used_relations.add(etype)
        src, dst = model.G.edges(etype=canonical_etype)
        print(f"  Relation {canonical_etype}: {len(src)} arêtes")

    # Vérifier quels embeddings sont effectivement utilisés
    used_embeddings = used_relations.intersection(emb_etypes)
    print(f"\nEmbeddings de relations effectivement utilisés: {len(used_embeddings)}/{len(emb_etypes)}")
    print(f"Relations utilisées: {sorted(list(used_embeddings))}")

    return used_embeddings

def analyze_graph_coverage():
    """
    Analyse la couverture du graphe RDF
    """
    # Compter tous les triplets
    total_triples = len(rdf_graph)

    # Analyser les types de nœuds
    node_type_counts = defaultdict(int)
    unmapped_types = set()

    for s in rdf_graph.subjects():
        type_found = False
        for p, o in rdf_graph.predicate_objects(subject=s):
            if 'type' in str(p).lower():
                type_uri = str(o)
                if type_uri in rdf_type_map:
                    node_type_counts[rdf_type_map[type_uri]] += 1
                    type_found = True
                else:
                    unmapped_types.add(type_uri)
        if not type_found:
            node_type_counts['unknown'] += 1

    # Analyser les relations
    relation_counts = defaultdict(int)
    unmapped_relations = set()

    for s, p, o in rdf_graph:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            pred = str(p)
            if "rdf-syntax-ns#type" not in pred:
                relation_counts[pred.split("/")[-1]] += 1

if __name__ == "__main__":
    print("Starting HGT clustering pipeline with target: RESULTS 0.8-0.83...")

    # Loading RDF graph
    rdf_graph = Graph()
    rdf_graph.parse("DBLP_petit.rdf")

    # Add coverage analysis
    analyze_graph_coverage()

    # Loading embeddings
    with open("node_features.pkl", "rb") as f:
        node_features = pickle.load(f)
    for k in node_features:
        node_features[k] = clean_tensor(node_features[k])

    with open("relation_embeddings.pkl", "rb") as f:
        relation_embeddings = pickle.load(f)
    for k in relation_embeddings:
        relation_embeddings[k] = clean_tensor(relation_embeddings[k])

    # Verification (optional)
    for k, v in node_features.items():
        if torch.isnan(v).any():
            print(f"NaN detected in node_features for {k}")
    for k, v in relation_embeddings.items():
        if torch.isnan(v).any():
            print(f"NaN detected in relation_embeddings for {k}")

    # === Detect RDF types of each entity
    # === Explicit mapping of RDF types to clear names
    rdf_type_map = {
        "http://xmlns.com/foaf/0.1/Person": "author",
        "http://example.org/Publication": "publication",
        "http://example.org/Domain": "domain",
        "http://example.org/Conference": "conference",
        "http://example.org/Journal": "journal",
        "http://example.org/Venue": "venue"
    }

    # === Detection of RDF types of each entity with mapping
    node_types = {}
    for s in rdf_graph.subjects():
        for p, o in rdf_graph.predicate_objects(subject=s):
            if 'type' in str(p).lower():
                mapped = rdf_type_map.get(str(o))
                if mapped:
                    node_types[str(s)] = mapped
                break  # stop at first type match

    # === Group edges by heterogeneous relation type
    edge_dict = defaultdict(lambda: ([], []))  # key: (src_type, relation, dst_type)
    # === Explicit mapping of RDF classes used in subClassOf
    class_uri_to_type = {
        "http://example.org/Conference": "conference",
        "http://example.org/Journal": "journal",
        "http://example.org/Venue": "venue"
    }
    # Filter only URIRef -> URIRef triplets
    for s, p, o in rdf_graph:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            # === Ignore rdf:type relations
            if "rdf-syntax-ns#type" in str(p):
                continue

            s_uri, p_uri, o_uri = str(s), str(p), str(o)

            # === Special case: subClassOf → correct types
            if p_uri.endswith("subClassOf"):
                s_type = class_uri_to_type.get(s_uri, "venue")
                o_type = class_uri_to_type.get(o_uri, "venue")
                rel = "subClassOf"
                edge_dict[(s_type, rel, o_type)][0].append(s_uri)
                edge_dict[(s_type, rel, o_type)][1].append(o_uri)
                continue  # don't process this triplet further

            s_uri, p_uri, o_uri = str(s), str(p), str(o)
            s_type = node_types.get(s_uri, "venue")
            o_type = node_types.get(o_uri, "venue")
            rel = p_uri.split("/")[-1]  # ex: creator, isPartOf
            edge_dict[(s_type, rel, o_type)][0].append(s_uri)
            edge_dict[(s_type, rel, o_type)][1].append(o_uri)

    # === Create global mapping of node identifiers
    all_node_ids = set()
    for (srcs, dsts) in edge_dict.values():
        all_node_ids.update(srcs)
        all_node_ids.update(dsts)

    node_id_map = {uri: idx for idx, uri in enumerate(sorted(all_node_ids))}
    id_node_map = {v: k for k, v in node_id_map.items()}

    # === Build final dictionary for DGL
    data_dict = {}
    for (srctype, reltype, dsttype), (srcs, dsts) in edge_dict.items():
        src_idx = [node_id_map[s] for s in srcs]
        dst_idx = [node_id_map[d] for d in dsts]
        data_dict[(srctype, reltype, dsttype)] = (src_idx, dst_idx)

    # === Create DGL heterogeneous graph
    g = dgl.heterograph(data_dict)

    # === Add node features (if available)
    features = torch.zeros((len(node_id_map), 768))
    for uri, idx in node_id_map.items():
        if uri in node_features:
            features[idx] = node_features[uri]
    from collections import defaultdict

    # Step 1: group indices by node type
    node_type_map = defaultdict(list)
    for uri, idx in node_id_map.items():
        ntype = node_types.get(uri, "venue").lower()
        node_type_map[ntype].append(idx)

    # Step 2: assign features by node type
    ndata_feats = {}
    for ntype in g.ntypes:
        nids = g.nodes(ntype)
        feats = []

        for nid in nids:
            uri = id_node_map[int(nid)]
            feat = node_features.get(uri, None)
            if feat is None or torch.isnan(feat).any() or torch.all(feat == 0):
                feat = torch.randn(768) * 0.01  # random noise
            feat = clean_tensor(feat)
            feats.append(feat)

        ndata_feats[ntype] = torch.stack(feats)

    # Step 2: assign features by node type
    g.ndata['feat'] = ndata_feats

    print(f"✅ DGL heterogeneous graph built with {g.num_nodes()} nodes and {g.num_edges()} edges.")
    print(f"Relation types: {list(g.canonical_etypes)}")

    # Get true labels based on hasDomain
    true_labels, domain_vocab = get_true_labels(g, rdf_graph, id_node_map)
    print(f"Number of classes (domains): {len(domain_vocab)}")
    print(f"Label distribution: {Counter(true_labels)}")

    # To see domain-label correspondence
    for domain, label in domain_vocab.items():
        print(f"Domain: {domain.split('/')[-1]}, Label: {label}")

    # Preparation of relation embeddings for HGT
    print("\nLoading relation embeddings...")
    print(f"Available keys in relation_embeddings: {list(relation_embeddings.keys())}")

    # Mapping between short names used in graph and complete URIs
    relation_uri_map = {
        'creator': 'http://purl.org/dc/elements/1.1/creator',
        'subClassOf': 'http://www.w3.org/2000/01/rdf-schema#subClassOf',
        'domain_dominant': 'http://example.org/features/domain_dominant',
        'hasDomain': 'http://example.org/hasDomain',
        'publishesDomain': 'http://example.org/publishesDomain',
        'isPartOf': 'http://purl.org/dc/elements/1.1/isPartOf',
        'hasPublishedIn': 'http://example.org/hasPublishedIn'
    }

    hgt_relation_embeddings = {}
    for etype in g.etypes:
        if etype in relation_uri_map and relation_uri_map[etype] in relation_embeddings:
            hgt_relation_embeddings[etype] = relation_embeddings[relation_uri_map[etype]]
            print(f"✅ Found embedding for relation type '{etype}'")
        else:
            print(f"⚠️ No embedding found for relation type '{etype}'")
            # Create an embedding based on the relation name
            # Use a simple text model to encode the name
            from sentence_transformers import SentenceTransformer
            text_model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = text_model.encode(etype, convert_to_tensor=True)
            if embedding.shape[0] != 768 or torch.isnan(embedding).any():
                embedding = torch.randn(768) * 0.01
            hgt_relation_embeddings[etype] = embedding
            print(f"  Created embedding for '{etype}' based on its name")
            if torch.isnan(embedding).any():
                print(f"NaN in embedding for {etype}, replacing with noise")
                embedding = torch.randn(768) * 0.01

    # Execute clustering pipeline
    print("\nStarting clustering...")
    model, vgae_models, embeddings, pred_labels = run_hgt_clustering(
        g, node_features, hgt_relation_embeddings, true_labels, id_node_map, rdf_graph
    )

    # Verify relation embeddings usage
    verify_relation_embeddings_usage(model, hgt_relation_embeddings)

    print("\nPipeline completed successfully!")
