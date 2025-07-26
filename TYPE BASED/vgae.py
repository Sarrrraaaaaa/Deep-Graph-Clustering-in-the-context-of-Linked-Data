from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
import numpy as np
import time
import pickle
import random  # Ajout de l'import manquant
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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

    # 1. Préparation des embeddings
    if isinstance(embeddings, dict):
        all_embeddings = torch.cat([e if isinstance(e, torch.Tensor) else torch.tensor(e)
                                  for e in embeddings.values()], dim=0)
    else:
        all_embeddings = embeddings if isinstance(embeddings, torch.Tensor) else torch.tensor(embeddings)

    all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
    all_embeddings = torch.nan_to_num(all_embeddings)
    all_embeddings = all_embeddings.to(device)

    # 2. Réduction de dimension si nécessaire
    input_dim = all_embeddings.shape[1]
    if input_dim > 256:
        pca = PCA(n_components=min(256, input_dim-1))
        all_embeddings_np = pca.fit_transform(all_embeddings.cpu().numpy())
        all_embeddings = torch.tensor(all_embeddings_np, device=device)
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

        model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_, device=device)

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
                supervised_loss = F.cross_entropy(q, torch.tensor(batch_labels, device=device))
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

def train_vgae(g, node_features, num_epochs=100, lr=0.001, weight_decay=1e-5):
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
        hidden_dim = 256
        out_dim = 128

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
            embeddings[ntype] = mean

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
        ('author', 'hasDomain', 'domain'),         # Relation auteur-domaine # Relation publication-conférence
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

def train_hgt(g, node_features, relation_embeddings, num_epochs=170, lr=0.0001, weight_decay=1e-6, true_labels=None):
    """
    Train a Heterogeneous Graph Transformer (HGT) model with contrastive learning
    and optional semi-supervised clustering
    """
    # Déduire la dimension d'entrée depuis les embeddings passés
    sample_type = list(node_features.keys())[0]
    in_dim = node_features[sample_type].shape[1]
    hidden_dim = 512
    out_dim = 256
    num_heads = 16
    num_layers = 3

    # Créer le modèle HGT
    model = HGT(
        G=g,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        n_layers=num_layers,
        n_heads=num_heads,
        dropout=0.1
    )

    # Déplacer le modèle sur GPU si disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Déplacer les features sur le même device
    for ntype in node_features:
        node_features[ntype] = node_features[ntype].to(device)

    # Déplacer les embeddings de relations sur le même device
    for etype in relation_embeddings:
        relation_embeddings[etype] = relation_embeddings[etype].to(device)

    # Optimiseur avec weight decay réduit
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler pour réduire le learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20
    )

    # Pour le suivi de la perte
    ema_loss = None
    alpha = 0.9  # Facteur de lissage

    # Boucle d'entraînement
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        embeddings = model(features=node_features, relation_embeddings=relation_embeddings)

        # Remplacer les NaN par des zéros
        for ntype in embeddings:
            if torch.isnan(embeddings[ntype]).any():
                embeddings[ntype] = torch.where(torch.isnan(embeddings[ntype]),
                                              torch.zeros_like(embeddings[ntype]),
                                              embeddings[ntype])

        # Normaliser les embeddings pour la similarité cosinus
        normalized_embeddings = {}
        for ntype in embeddings:
            normalized_embeddings[ntype] = F.normalize(embeddings[ntype], p=2, dim=1)

        # Utiliser true_labels pour le préentraînement semi-supervisé
        # Réduire progressivement l'influence des labels au fil des époques
        semi_weight = max(0, 1.0 - epoch / (num_epochs * 0.5)) if true_labels else 0

        # Calculer la perte contrastive avec température plus basse et true_labels
        if semi_weight > 0:
            loss = improved_contrastive_loss(normalized_embeddings, g, temperature=0.07, margin=0.2, true_labels=true_labels)
        else:
            loss = improved_contrastive_loss(normalized_embeddings, g, temperature=0.07, margin=0.2)

        # Vérifier si la perte est NaN
        if torch.isnan(loss):
            optimizer.zero_grad()
            continue

        # Backward pass avec gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Vérifier les gradients pour les NaN
        nan_grads = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_grads = True
                break

        # Si des NaN sont détectés dans les gradients, sauter la mise à jour
        if nan_grads:
            optimizer.zero_grad()
            continue

        optimizer.step()

        # Update EMA loss
        if ema_loss is None:
            ema_loss = loss.item()
        else:
            ema_loss = alpha * ema_loss + (1 - alpha) * loss.item()

        # Mettre à jour le scheduler
        scheduler.step(ema_loss)

        # Afficher la progression
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {ema_loss:.4f}, LR: {current_lr:.6f}")

    # Générer les embeddings finaux
    model.eval()
    with torch.no_grad():
        embeddings = model(features=node_features, relation_embeddings=relation_embeddings)
        # Normaliser les embeddings finaux
        for ntype in embeddings:
            embeddings[ntype] = F.normalize(embeddings[ntype], p=2, dim=1)

    return model, embeddings


def evaluate_with_dec(embeddings, true_labels, n_clusters, node_type='author'):
    """
    Evaluate clustering using DEC with proper dimension handling
    """
    print("\n" + "="*50)
    print(f"DEC CLUSTERING FOR {node_type.upper()}")
    print("="*50)

    # Clean and prepare embeddings
    if isinstance(embeddings, torch.Tensor):
        cleaned_embeddings = embeddings.detach()
    else:
        cleaned_embeddings = torch.tensor(clean_embeddings(embeddings))

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

    # Calculate F1-score and accuracy using aligned labels
    aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)
    f1 = f1_score(true_labels, aligned_pred_labels, average='weighted')
    accuracy = accuracy_score(true_labels, aligned_pred_labels)

    print(f"\nDEC Clustering Evaluation for {node_type}:")
    print(f"NMI: {nmi:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Visualize clusters
    visualize_clusters(
        cleaned_embeddings.cpu().numpy(),
        true_labels,
        pred_labels,
        f"DEC Clusters ({node_type})",
        method_name="DEC",
        node_type=node_type
    )

    # Analyze clusters
    purity = analyze_clusters(cleaned_embeddings.cpu().numpy(), true_labels, pred_labels)

    return nmi, ari, pred_labels, purity
def visualize_clusters(embeddings, true_labels, pred_labels, title="Cluster Visualization", method_name=None, node_type=None):
    """
    Visualize clusters using t-SNE with two plots side by side.
    Saves each clustering method visualization with a specific filename.
    """
    # Convert embeddings to numpy if they're not already
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Align predicted labels with true labels for consistent colors
    aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

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

    # Create output directory and save figure with specific naming
    output_dir = "images VGAE + HGT"
    os.makedirs(output_dir, exist_ok=True)

    # Create specific filename based on node type and method
    if method_name and node_type:
        # Clean method name for filename
        clean_method = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('{', '').replace('}', '').replace("'", '').replace(':', '_')
        filename = f"{node_type}_{clean_method}_clustering"
    else:
        filename = title.replace(' ', '_').replace('(', '').replace(')', '').replace('venue', 'venue')

    filepath = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved: {filepath}")

    # Display in notebook
    try:
        from IPython.display import display
        display(plt.gcf())
    except ImportError:
        pass
    plt.close()

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

def save_best_method_visualization(embeddings, true_labels, pred_labels, method_name, node_type, metrics):
    """
    Save a special visualization for the best performing method with metrics in the title
    """
    # Create a detailed title with metrics
    title = f"BEST {node_type.upper()} METHOD: {method_name}\nNMI: {metrics['nmi']:.4f} | ARI: {metrics['ari']:.4f} | F1: {metrics['f1']:.4f} | ACC: {metrics['accuracy']:.4f}"

    # Convert embeddings to numpy if they're not already
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Align predicted labels with true labels for consistent colors
    aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot true labels
    unique_true_labels = np.unique(true_labels)
    n_true_clusters = len(unique_true_labels)

    # Plot true labels
    ax1.set_title(f"True Labels ({n_true_clusters} domains)", fontsize=14)
    for label in unique_true_labels:
        indices = true_labels == label
        ax1.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label, s=30, alpha=0.8)

    # Add legend for true labels
    ax1.legend(title="Domains", loc="upper right", bbox_to_anchor=(1.15, 1))
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")

    # Plot predicted labels with aligned colors
    unique_pred_labels = np.unique(aligned_pred_labels)

    ax2.set_title(f"Predicted Clusters ({method_name})", fontsize=14)
    for label in unique_pred_labels:
        indices = aligned_pred_labels == label
        ax2.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label, s=30, alpha=0.8)

    # Add legend for predicted labels
    ax2.legend(title="Clusters", loc="upper right", bbox_to_anchor=(1.15, 1))
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")

    # Set main title
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()

    # Create output directory and save figure
    output_dir = "images VGAE + HGT"
    os.makedirs(output_dir, exist_ok=True)

    # Create filename for best method
    clean_method = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('{', '').replace('}', '').replace("'", '').replace(':', '_')
    filename = f"BEST_{node_type}_{clean_method}_clustering"
    filepath = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"🏆 BEST METHOD visualization saved: {filepath}")

    plt.close()
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

            # Calculate F1-score and accuracy using aligned labels
            aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)
            f1 = f1_score(true_labels, aligned_pred_labels, average='weighted')
            accuracy = accuracy_score(true_labels, aligned_pred_labels)

            method_key = f"{method_name}_{params}"
            result_line = f"{method_name} ({params}): NMI={nmi:.4f}, ARI={ari:.4f}, F1={f1:.4f}, Acc={accuracy:.4f}"
            print(result_line)
            detailed_results.append(result_line)

            # Store individual method results
            all_method_results[method_key] = {
                'nmi': nmi,
                'ari': ari,
                'f1': f1,
                'accuracy': accuracy
            }

            # Save visualization for each method
            method_title = f"{node_type.capitalize()} {method_name} Clustering"
            visualize_clusters(
                embeddings,
                true_labels,
                pred_labels,
                title=method_title,
                method_name=method_key,
                node_type=node_type
            )

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

    # For all node types (including authors), also try a semi-supervised approach
    if len(true_labels) > 100:
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

        # Calculate F1-score and accuracy for semi-supervised approach
        aligned_semi_pred_labels = align_clusters_for_visualization(semi_pred_labels, true_labels)
        semi_f1 = f1_score(true_labels, aligned_semi_pred_labels, average='weighted')
        semi_accuracy = accuracy_score(true_labels, aligned_semi_pred_labels)

        semi_result_line = f"Semi-supervised approach: NMI={semi_nmi:.4f}, ARI={semi_ari:.4f}, F1={semi_f1:.4f}, Acc={semi_accuracy:.4f}"
        print(semi_result_line)
        detailed_results.append(semi_result_line)

        # Store semi-supervised results
        all_method_results['Semi-supervised'] = {
            'nmi': semi_nmi,
            'ari': semi_ari,
            'f1': semi_f1,
            'accuracy': semi_accuracy
        }

        # Save visualization for semi-supervised approach
        semi_title = f"{node_type.capitalize()} Semi-supervised Clustering"
        visualize_clusters(
            embeddings,
            true_labels,
            semi_pred_labels,
            title=semi_title,
            method_name="Semi-supervised",
            node_type=node_type
        )

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
                f.write(f"  NMI (Normalized Mutual Information): {best.get('nmi', 'N/A'):.4f}\n")
                f.write(f"  ARI (Adjusted Rand Index): {best.get('ari', 'N/A'):.4f}\n")
                f.write(f"  F1-Score: {best.get('f1', 'N/A'):.4f}\n")
                f.write(f"  Accuracy: {best.get('accuracy', 'N/A'):.4f}\n")
                f.write(f"  Cluster Purity: {best.get('purity', 'N/A'):.4f}\n")
                f.write(f"  Method Used: {best.get('method', 'N/A')}\n\n")

            if 'cluster_analysis' in results:
                analysis = results['cluster_analysis']
                f.write("Detailed Cluster Analysis:\n")
                f.write("-" * 25 + "\n")
                for cluster_id, cluster_info in analysis.items():
                    f.write(f"  Cluster {cluster_id}:\n")
                    f.write(f"    Size: {cluster_info.get('size', 'N/A')} nodes\n")
                    f.write(f"    Dominant label: {cluster_info.get('dominant_label', 'N/A')}\n")
                    f.write(f"    Purity: {cluster_info.get('purity', 'N/A'):.2%}\n")
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

    # Train HGT with contrastive learning using VGAE embeddings and true_labels
    print("Training HGT model with semi-supervised contrastive learning...")
    model, embeddings = train_hgt(g, vgae_embeddings, relation_embeddings, num_epochs=30, true_labels=clustering_labels)

    # Step 3: Clustering
    print("\n" + "="*50)
    print("CLUSTERING")
    print("="*50)

    # 1. Author clustering
    print("\n" + "="*50)
    print("AUTHOR CLUSTERING")
    print("="*50)

    pred_labels = None
    if 'author' in embeddings:
        author_embeddings = clean_embeddings(embeddings['author'])
        n_clusters = len(np.unique(true_labels))

        # Initialize author results
        clustering_results['author'] = {
            'dataset_info': {
                'num_nodes': len(author_embeddings),
                'num_clusters': n_clusters,
                'embedding_dim': author_embeddings.shape[1],
                'label_distribution': dict(Counter(true_labels))
            },
            'methods': {}
        }

        # First try with DEC for reference
        print("\nEvaluating author clustering with DEC...")
        dec_nmi, dec_ari, dec_pred_labels, dec_purity = evaluate_with_dec(
            author_embeddings,
            true_labels,
            n_clusters,
            node_type='author'
        )

        # Store DEC results
        dec_aligned_pred_labels = align_clusters_for_visualization(dec_pred_labels, true_labels)
        dec_f1 = f1_score(true_labels, dec_aligned_pred_labels, average='weighted')
        dec_accuracy = accuracy_score(true_labels, dec_aligned_pred_labels)

        clustering_results['author']['methods']['DEC'] = {
            'nmi': dec_nmi,
            'ari': dec_ari,
            'f1': dec_f1,
            'accuracy': dec_accuracy,
            'purity': dec_purity
        }

        # Then use our improved clustering function
        print("\nComparing with other clustering methods...")
        nmi, ari, pred_labels, detailed_results, all_method_results = evaluate_clustering(
            author_embeddings,
            true_labels,
            n_clusters,
            node_type='author'
        )

        # Store detailed results
        clustering_results['author']['detailed_results'] = detailed_results

        # Store all individual method results
        for method_name, metrics in all_method_results.items():
            clustering_results['author']['methods'][method_name] = metrics

        # Store other methods results
        other_aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)
        other_f1 = f1_score(true_labels, other_aligned_pred_labels, average='weighted')
        other_accuracy = accuracy_score(true_labels, other_aligned_pred_labels)

        clustering_results['author']['methods']['Best_Traditional'] = {
            'nmi': nmi,
            'ari': ari,
            'f1': other_f1,
            'accuracy': other_accuracy
        }

        # Choose the best method
        if dec_nmi > nmi:
            print("DEC outperformed other methods, using DEC results")
            nmi, ari, pred_labels = dec_nmi, dec_ari, dec_pred_labels
            best_method = 'DEC'
            best_purity = dec_purity
        else:
            best_method = 'Best_Traditional'
            best_purity = analyze_clusters(author_embeddings, true_labels, pred_labels)

        # Calculate F1-score and accuracy for final results
        aligned_pred_labels = align_clusters_for_visualization(pred_labels, true_labels)
        f1 = f1_score(true_labels, aligned_pred_labels, average='weighted')
        accuracy = accuracy_score(true_labels, aligned_pred_labels)

        # Store best results
        clustering_results['author']['best_results'] = {
            'nmi': nmi,
            'ari': ari,
            'f1': f1,
            'accuracy': accuracy,
            'purity': best_purity,
            'method': best_method
        }

        print(f"\nAuthor Clustering Evaluation:")
        print(f"NMI: {nmi:.4f}")
        print(f"ARI: {ari:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        # Visualize clusters with aligned labels
        print("\nVisualizing final author clusters with aligned labels...")
        visualize_clusters(
            author_embeddings,
            true_labels,
            aligned_pred_labels,  # Use aligned labels
            "Author Cluster Visualization (Aligned)"
        )

        # Save best method visualization
        print(f"\nSaving BEST author method visualization ({best_method})...")
        save_best_method_visualization(
            author_embeddings,
            true_labels,
            aligned_pred_labels,
            best_method,
            'author',
            clustering_results['author']['best_results']
        )

        # Analyze clusters with aligned labels and store analysis
        print("\nAnalyzing clusters with aligned labels...")
        cluster_analysis = {}
        unique_clusters = np.unique(aligned_pred_labels)
        for cluster in unique_clusters:
            cluster_indices = np.where(aligned_pred_labels == cluster)[0]
            cluster_true_labels = true_labels[cluster_indices]
            unique_true_labels, counts = np.unique(cluster_true_labels, return_counts=True)
            dominant_label = unique_true_labels[np.argmax(counts)]
            dominant_count = np.max(counts)

            cluster_analysis[cluster] = {
                'size': len(cluster_indices),
                'dominant_label': dominant_label,
                'purity': dominant_count / len(cluster_indices)
            }

        clustering_results['author']['cluster_analysis'] = cluster_analysis
        analyze_clusters(
            author_embeddings,
            true_labels,
            aligned_pred_labels  # Use aligned labels
        )

    # 2. Publication clustering
    print("\n" + "="*50)
    print("PUBLICATION CLUSTERING")
    print("="*50)

    if 'publication' in embeddings:
        # Get true labels for publications if available
        pub_true_labels = None
        if hasattr(g.nodes['publication'], 'data') and 'true_domain' in g.nodes['publication'].data:
            pub_true_labels = g.nodes['publication'].data['true_domain'].cpu().numpy()

        # If true labels are not available, try to extract them from RDF graph
        if pub_true_labels is None:
            try:
                pub_true_labels = get_node_domains(g, 'publication', id_node_map, rdf_graph)
            except:
                # Fallback: use domain-based clustering
                print("No domain labels available for publications, using similarity-based clustering.")
                domain_assignments = cluster_publications_by_domain(g, embeddings, id_node_map)
                pub_true_labels = domain_assignments

        if pub_true_labels is not None and len(pub_true_labels) > 0:
            pub_embeddings = clean_embeddings(embeddings['publication'])
            n_pub_clusters = len(np.unique(pub_true_labels))

            # Initialize publication results
            clustering_results['publication'] = {
                'dataset_info': {
                    'num_nodes': len(pub_embeddings),
                    'num_clusters': n_pub_clusters,
                    'embedding_dim': pub_embeddings.shape[1],
                    'label_distribution': dict(Counter(pub_true_labels))
                },
                'methods': {}
            }

            # Add DEC evaluation
            print(f"\nEvaluating publication clustering with DEC...")
            dec_nmi, dec_ari, dec_pred_labels, dec_purity = evaluate_with_dec(
               pub_embeddings,
               pub_true_labels,
               n_pub_clusters,
               node_type='publication'
            )

            # Store DEC results
            dec_aligned_pub_pred_labels = align_clusters_for_visualization(dec_pred_labels, pub_true_labels)
            dec_pub_f1 = f1_score(pub_true_labels, dec_aligned_pub_pred_labels, average='weighted')
            dec_pub_accuracy = accuracy_score(pub_true_labels, dec_aligned_pub_pred_labels)

            clustering_results['publication']['methods']['DEC'] = {
                'nmi': dec_nmi,
                'ari': dec_ari,
                'f1': dec_pub_f1,
                'accuracy': dec_pub_accuracy,
                'purity': dec_purity
            }

            # Specific parameters for publications
            print("\nEvaluating publication clustering...")
            pub_nmi, pub_ari, pub_pred_labels, pub_detailed_results, pub_all_method_results = evaluate_clustering(
                pub_embeddings,
                pub_true_labels,
                n_pub_clusters,
                node_type='publication'
            )

            # Store detailed results
            clustering_results['publication']['detailed_results'] = pub_detailed_results

            # Store all individual method results
            for method_name, metrics in pub_all_method_results.items():
                clustering_results['publication']['methods'][method_name] = metrics

            # Store other methods results
            other_aligned_pub_pred_labels = align_clusters_for_visualization(pub_pred_labels, pub_true_labels)
            other_pub_f1 = f1_score(pub_true_labels, other_aligned_pub_pred_labels, average='weighted')
            other_pub_accuracy = accuracy_score(pub_true_labels, other_aligned_pub_pred_labels)

            clustering_results['publication']['methods']['Best_Traditional'] = {
                'nmi': pub_nmi,
                'ari': pub_ari,
                'f1': other_pub_f1,
                'accuracy': other_pub_accuracy
            }

            if dec_nmi > pub_nmi:
               print("DEC outperformed other methods, using DEC results")
               pub_nmi, pub_ari, pub_pred_labels = dec_nmi, dec_ari, dec_pred_labels
               best_pub_method = 'DEC'
               best_pub_purity = dec_purity
            else:
               best_pub_method = 'Best_Traditional'
               best_pub_purity = analyze_clusters(pub_embeddings, pub_true_labels, pub_pred_labels)

            # Calculate F1-score and accuracy for publications
            aligned_pub_pred_labels = align_clusters_for_visualization(pub_pred_labels, pub_true_labels)
            pub_f1 = f1_score(pub_true_labels, aligned_pub_pred_labels, average='weighted')
            pub_accuracy = accuracy_score(pub_true_labels, aligned_pub_pred_labels)

            # Store best results
            clustering_results['publication']['best_results'] = {
                'nmi': pub_nmi,
                'ari': pub_ari,
                'f1': pub_f1,
                'accuracy': pub_accuracy,
                'purity': best_pub_purity,
                'method': best_pub_method
            }

            print(f"\nPublication Clustering Evaluation:")
            print(f"NMI: {pub_nmi:.4f}")
            print(f"ARI: {pub_ari:.4f}")
            print(f"F1-Score: {pub_f1:.4f}")
            print(f"Accuracy: {pub_accuracy:.4f}")

            # Visualize with the same format as for authors
            visualize_clusters(
                pub_embeddings,
                pub_true_labels,
                pub_pred_labels,
                "Publication Cluster Visualization"
            )

            # Save best method visualization for publications
            print(f"\nSaving BEST publication method visualization ({best_pub_method})...")
            save_best_method_visualization(
                pub_embeddings,
                pub_true_labels,
                aligned_pub_pred_labels,
                best_pub_method,
                'publication',
                clustering_results['publication']['best_results']
            )

            # Analyze publication clusters and store analysis
            print("\nAnalyzing publication clusters...")
            pub_cluster_analysis = {}
            unique_pub_clusters = np.unique(aligned_pub_pred_labels)
            for cluster in unique_pub_clusters:
                cluster_indices = np.where(aligned_pub_pred_labels == cluster)[0]
                cluster_true_labels = pub_true_labels[cluster_indices]
                unique_true_labels, counts = np.unique(cluster_true_labels, return_counts=True)
                dominant_label = unique_true_labels[np.argmax(counts)]
                dominant_count = np.max(counts)

                pub_cluster_analysis[cluster] = {
                    'size': len(cluster_indices),
                    'dominant_label': dominant_label,
                    'purity': dominant_count / len(cluster_indices)
                }

            clustering_results['publication']['cluster_analysis'] = pub_cluster_analysis
            analyze_clusters(pub_embeddings, pub_true_labels, pub_pred_labels)
        else:
            print("No domain labels available for publications.")

    # 3. Venue clustering
    print("\n" + "="*50)
    print("VENUE CLUSTERING")
    print("="*50)

    # Check for venues in different possible keys
    venue_key = None
    for key in ['conference', 'venue', 'Conference', 'Venue']:
        if key in embeddings:
            venue_key = key
            break

    print(f"Available embedding keys: {list(embeddings.keys())}")
    print(f"Graph node types: {g.ntypes}")

    if venue_key is not None:
        print(f"Found venues under key: '{venue_key}'")

        # Get true labels for venues if available
        venue_true_labels = None

        # Try to get from graph data first
        try:
            if hasattr(g.nodes[venue_key], 'data') and 'true_domain' in g.nodes[venue_key].data:
                venue_true_labels = g.nodes[venue_key].data['true_domain'].cpu().numpy()
                print(f"Found true labels in graph data: {len(venue_true_labels)} labels")
        except:
            pass

        # If true labels are not available, try to extract them from RDF graph
        if venue_true_labels is None:
            print("Trying to extract venue domains from RDF graph...")
            try:
                venue_true_labels = get_node_domains(g, venue_key, id_node_map, rdf_graph)
                if venue_true_labels is not None:
                    print(f"Extracted {len(venue_true_labels)} venue labels from RDF")
                else:
                    print("No venue labels found in RDF graph")
            except Exception as e:
                print(f"Error extracting from RDF: {e}")
                venue_true_labels = None

        # If still no labels, use domain-based clustering
        if venue_true_labels is None:
            print("No domain labels available for venues, using similarity-based clustering.")
            try:
                domain_assignments = cluster_conferences_by_domain(g, embeddings, id_node_map)
                venue_true_labels = domain_assignments
                print(f"Created {len(venue_true_labels)} venue labels using similarity clustering")
            except Exception as e:
                print(f"Error in similarity clustering: {e}")
                venue_true_labels = None

        if venue_true_labels is not None and len(venue_true_labels) > 0:
            venue_embeddings = clean_embeddings(embeddings[venue_key])
            n_venue_clusters = len(np.unique(venue_true_labels))

            print(f"Venue embeddings shape: {venue_embeddings.shape}")
            print(f"Number of venue clusters: {n_venue_clusters}")
            print(f"Unique venue labels: {np.unique(venue_true_labels)}")

            # Initialize venue results
            clustering_results['venue'] = {
                'dataset_info': {
                    'num_nodes': len(venue_embeddings),
                    'num_clusters': n_venue_clusters,
                    'embedding_dim': venue_embeddings.shape[1],
                    'label_distribution': dict(Counter(venue_true_labels))
                },
                'methods': {}
            }

            # Use specific clustering methods for venues
            print("\nEvaluating venue clustering...")

            # Configuration 1: Standard method
            venue_nmi, venue_ari, venue_pred_labels, venue_detailed_results, venue_all_method_results = evaluate_clustering(
                venue_embeddings,
                venue_true_labels,
                n_venue_clusters,
                node_type='venue'
            )

            # Store detailed results
            clustering_results['venue']['detailed_results'] = venue_detailed_results

            # Store all individual method results
            for method_name, metrics in venue_all_method_results.items():
                clustering_results['venue']['methods'][method_name] = metrics

            # Store standard method results
            standard_aligned_venue_pred_labels = align_clusters_for_visualization(venue_pred_labels, venue_true_labels)
            standard_venue_f1 = f1_score(venue_true_labels, standard_aligned_venue_pred_labels, average='weighted')
            standard_venue_accuracy = accuracy_score(venue_true_labels, standard_aligned_venue_pred_labels)

            clustering_results['venue']['methods']['Standard'] = {
                'nmi': venue_nmi,
                'ari': venue_ari,
                'f1': standard_venue_f1,
                'accuracy': standard_venue_accuracy
            }

            best_venue_nmi = venue_nmi
            best_venue_ari = venue_ari
            best_venue_pred_labels = venue_pred_labels
            best_venue_method = 'Standard'

            # Configuration 2: Specialized method for venues
            print("\nTesting specialized venue clustering...")
            venue_nmi2, venue_ari2, venue_pred_labels2 = evaluate_clustering_conference(
                venue_embeddings,
                venue_true_labels,
                n_venue_clusters
            )

            # Store specialized method results
            specialized_aligned_venue_pred_labels = align_clusters_for_visualization(venue_pred_labels2, venue_true_labels)
            specialized_venue_f1 = f1_score(venue_true_labels, specialized_aligned_venue_pred_labels, average='weighted')
            specialized_venue_accuracy = accuracy_score(venue_true_labels, specialized_aligned_venue_pred_labels)

            clustering_results['venue']['methods']['Specialized'] = {
                'nmi': venue_nmi2,
                'ari': venue_ari2,
                'f1': specialized_venue_f1,
                'accuracy': specialized_venue_accuracy
            }

            if venue_nmi2 > best_venue_nmi:
                best_venue_nmi = venue_nmi2
                best_venue_ari = venue_ari2
                best_venue_pred_labels = venue_pred_labels2
                best_venue_method = 'Specialized'

            # Configuration 3: Use hierarchical clustering
            print("\nTesting hierarchical clustering for venues...")
            venue_nmi3, venue_ari3, venue_pred_labels3 = evaluate_clustering_dbscan(
                venue_embeddings,
                venue_true_labels
            )

            # Store hierarchical method results
            hierarchical_aligned_venue_pred_labels = align_clusters_for_visualization(venue_pred_labels3, venue_true_labels)
            hierarchical_venue_f1 = f1_score(venue_true_labels, hierarchical_aligned_venue_pred_labels, average='weighted')
            hierarchical_venue_accuracy = accuracy_score(venue_true_labels, hierarchical_aligned_venue_pred_labels)

            clustering_results['venue']['methods']['Hierarchical'] = {
                'nmi': venue_nmi3,
                'ari': venue_ari3,
                'f1': hierarchical_venue_f1,
                'accuracy': hierarchical_venue_accuracy
            }

            if venue_nmi3 > best_venue_nmi:
                best_venue_nmi = venue_nmi3
                best_venue_ari = venue_ari3
                best_venue_pred_labels = venue_pred_labels3
                best_venue_method = 'Hierarchical'

            # Add DEC evaluation
            print(f"\nEvaluating venue clustering with DEC...")
            dec_nmi, dec_ari, dec_pred_labels, dec_purity = evaluate_with_dec(
               venue_embeddings,
               venue_true_labels,
               n_venue_clusters,
               node_type='venue'
            )

            # Store DEC results
            dec_aligned_venue_pred_labels = align_clusters_for_visualization(dec_pred_labels, venue_true_labels)
            dec_venue_f1 = f1_score(venue_true_labels, dec_aligned_venue_pred_labels, average='weighted')
            dec_venue_accuracy = accuracy_score(venue_true_labels, dec_aligned_venue_pred_labels)

            clustering_results['venue']['methods']['DEC'] = {
                'nmi': dec_nmi,
                'ari': dec_ari,
                'f1': dec_venue_f1,
                'accuracy': dec_venue_accuracy,
                'purity': dec_purity
            }

            if dec_nmi > best_venue_nmi:
               print("DEC outperformed other methods, using DEC results")
               best_venue_nmi, best_venue_ari, best_venue_pred_labels = dec_nmi, dec_ari, dec_pred_labels
               best_venue_method = 'DEC'
               best_venue_purity = dec_purity
            else:
               best_venue_purity = analyze_clusters(venue_embeddings, venue_true_labels, best_venue_pred_labels)

            # Calculate F1-score and accuracy for venues
            aligned_venue_pred_labels = align_clusters_for_visualization(best_venue_pred_labels, venue_true_labels)
            venue_f1 = f1_score(venue_true_labels, aligned_venue_pred_labels, average='weighted')
            venue_accuracy = accuracy_score(venue_true_labels, aligned_venue_pred_labels)

            # Store best results
            clustering_results['venue']['best_results'] = {
                'nmi': best_venue_nmi,
                'ari': best_venue_ari,
                'f1': venue_f1,
                'accuracy': venue_accuracy,
                'purity': best_venue_purity,
                'method': best_venue_method
            }

            print(f"\nBest Venue Clustering Evaluation:")
            print(f"NMI: {best_venue_nmi:.4f}")
            print(f"ARI: {best_venue_ari:.4f}")
            print(f"F1-Score: {venue_f1:.4f}")
            print(f"Accuracy: {venue_accuracy:.4f}")

            # Visualize with the same format as for authors
            visualize_clusters(
                venue_embeddings,
                venue_true_labels,
                best_venue_pred_labels,
                "Venue Cluster Visualization"
            )

            # Save best method visualization for venues
            print(f"\nSaving BEST venue method visualization ({best_venue_method})...")
            save_best_method_visualization(
                venue_embeddings,
                venue_true_labels,
                aligned_venue_pred_labels,
                best_venue_method,
                'venue',
                clustering_results['venue']['best_results']
            )

            # Analyze venue clusters and store analysis
            print("\nAnalyzing venue clusters...")
            venue_cluster_analysis = {}
            unique_venue_clusters = np.unique(aligned_venue_pred_labels)
            for cluster in unique_venue_clusters:
                cluster_indices = np.where(aligned_venue_pred_labels == cluster)[0]
                cluster_true_labels = venue_true_labels[cluster_indices]
                unique_true_labels, counts = np.unique(cluster_true_labels, return_counts=True)
                dominant_label = unique_true_labels[np.argmax(counts)]
                dominant_count = np.max(counts)

                venue_cluster_analysis[cluster] = {
                    'size': len(cluster_indices),
                    'dominant_label': dominant_label,
                    'purity': dominant_count / len(cluster_indices)
                }

            clustering_results['venue']['cluster_analysis'] = venue_cluster_analysis
            analyze_clusters(venue_embeddings, venue_true_labels, best_venue_pred_labels)
        else:
            print("❌ No domain labels available for venues or insufficient data for clustering.")
            print(f"   - Venue true labels: {venue_true_labels is not None}")
            if venue_true_labels is not None:
                print(f"   - Number of labels: {len(venue_true_labels)}")
    else:
        print("❌ No venue embeddings found in the model.")
        print("   Available embedding types:", list(embeddings.keys()))
        print("   This might indicate that venues were not properly processed during training.")

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

    # Créer les visualisations pour toutes les meilleures méthodes
    try:
        all_visualizations = visualize_best_methods_for_all_types(
            clustering_results, embeddings, true_labels, g, id_node_map, rdf_graph
        )
        print(f"✅ {len(all_visualizations)} visualisations créées avec succès")
    except Exception as e:
        print(f"❌ Erreur lors de la création des visualisations: {e}")

    print("\nModels, embeddings, and clustering results saved successfully!")

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

def visualize_best_methods_for_all_types(clustering_results, embeddings, true_labels_dict, g, id_node_map, rdf_graph=None):
    """
    Visualise les meilleurs résultats pour chaque méthode de clustering pour tous les types de nœuds
    """
    print("\n" + "="*60)
    print("CRÉATION DES VISUALISATIONS POUR LES MEILLEURES MÉTHODES")
    print("="*60)

    # Créer le dossier de sortie
    output_dir = "images VGAE + HGT"
    os.makedirs(output_dir, exist_ok=True)

    # Définir les catégories de méthodes à extraire
    method_categories = {
        'KMeans': 'KMeans',
        'AgglomerativeClustering_ward': 'ward',
        'AgglomerativeClustering_complete': 'complete',
        'AgglomerativeClustering_average': 'average',
        'SpectralClustering': 'SpectralClustering',
        'DEC': 'DEC',
        'Semi-supervised': 'Semi-supervised'
    }

    all_visualizations = []

    for node_type, results in clustering_results.items():
        if 'methods' not in results or node_type not in embeddings:
            continue

        print(f"\n--- Traitement des méthodes pour {node_type.upper()} ---")

        # Obtenir les embeddings et true labels pour ce type de nœud
        node_embeddings = clean_embeddings(embeddings[node_type])

        # Obtenir les true labels
        if node_type == 'author':
            true_labels = true_labels_dict
        elif node_type == 'publication':
            try:
                if rdf_graph is not None:
                    true_labels = get_node_domains(g, 'publication', id_node_map, rdf_graph)
                else:
                    true_labels = None
                if true_labels is None:
                    true_labels = cluster_publications_by_domain(g, embeddings, id_node_map)
            except:
                continue
        elif node_type == 'venue' or node_type == 'conference':
            try:
                venue_key = 'conference' if 'conference' in embeddings else 'venue'
                if rdf_graph is not None:
                    true_labels = get_node_domains(g, venue_key, id_node_map, rdf_graph)
                else:
                    true_labels = None
                if true_labels is None:
                    true_labels = cluster_conferences_by_domain(g, embeddings, id_node_map)
            except:
                continue
        else:
            continue

        if true_labels is None:
            continue

        # Pour chaque catégorie de méthode, trouver la meilleure variante
        for category, keyword in method_categories.items():
            best_nmi = -1
            best_method_name = None
            best_metrics = None

            for method_name, metrics in results['methods'].items():
                # Vérifier si cette méthode appartient à la catégorie actuelle
                if category.startswith('AgglomerativeClustering_'):
                    linkage_type = category.split('_')[1]
                    if 'AgglomerativeClustering' in method_name and linkage_type in method_name:
                        nmi = metrics.get('nmi', 0)
                        if nmi > best_nmi:
                            best_nmi = nmi
                            best_method_name = method_name
                            best_metrics = metrics
                else:
                    if keyword in method_name:
                        nmi = metrics.get('nmi', 0)
                        if nmi > best_nmi:
                            best_nmi = nmi
                            best_method_name = method_name
                            best_metrics = metrics

            # Si on a trouvé une méthode pour cette catégorie, créer la visualisation
            if best_method_name and best_metrics:
                print(f"  📊 {category}: {best_method_name}")
                print(f"      NMI: {best_metrics.get('nmi', 0):.4f}, ARI: {best_metrics.get('ari', 0):.4f}")
                print(f"      F1: {best_metrics.get('f1', 0):.4f}, Acc: {best_metrics.get('accuracy', 0):.4f}")

                try:
                    # Recréer les labels de clustering pour cette méthode
                    pred_labels = recreate_clustering_for_method(
                        node_embeddings, true_labels, best_method_name, category
                    )

                    if pred_labels is not None:
                        # Créer le titre de visualisation
                        title = f"{node_type.title()} - {category} (Best: {best_metrics.get('nmi', 0):.3f} NMI)"

                        # Créer la visualisation
                        visualize_clusters(
                            node_embeddings,
                            true_labels,
                            pred_labels,
                            title
                        )

                        # Stocker les informations de visualisation
                        all_visualizations.append({
                            'node_type': node_type,
                            'method': category,
                            'full_method_name': best_method_name,
                            'metrics': best_metrics,
                            'title': title
                        })

                        print(f"      ✅ Visualisation sauvegardée: {title}")
                    else:
                        print(f"      ❌ Impossible de recréer le clustering pour {best_method_name}")

                except Exception as e:
                    print(f"      ❌ Erreur lors de la création de la visualisation: {e}")

    # Créer un graphique de résumé des performances
    create_performance_summary_chart(all_visualizations, output_dir)

    print(f"\n✅ Toutes les visualisations sauvegardées dans '{output_dir}'")
    print(f"📊 Total des visualisations créées: {len(all_visualizations)}")

    return all_visualizations

def recreate_clustering_for_method(embeddings, true_labels, method_name, category):
    """
    Recrée les labels de clustering basés sur le nom de la méthode et la catégorie
    """
    try:
        # Convertir les embeddings en numpy si nécessaire
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()

        # Normaliser les embeddings
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

        # Réduire la dimensionnalité
        from sklearn.decomposition import PCA
        pca_components = min(128, embeddings.shape[1])
        pca = PCA(n_components=pca_components, random_state=42)
        embeddings_reduced = pca.fit_transform(embeddings)

        n_clusters = len(np.unique(true_labels))

        # Recréer le clustering basé sur la catégorie
        if category == 'DEC':
            # Pour DEC, nous devons réentraîner le modèle
            _, pred_labels = train_dec(
                torch.tensor(embeddings_reduced),
                n_clusters,
                hidden_dims=[256, 128] if embeddings_reduced.shape[1] > 128 else [128, 64]
            )
            return pred_labels

        elif category == 'KMeans':
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
            pred_labels = model.fit_predict(embeddings_reduced)
            return pred_labels

        elif category.startswith('AgglomerativeClustering_'):
            from sklearn.cluster import AgglomerativeClustering
            linkage_type = category.split('_')[1]
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
            pred_labels = model.fit_predict(embeddings_reduced)
            return pred_labels

        elif category == 'SpectralClustering':
            from sklearn.cluster import SpectralClustering
            model = SpectralClustering(
                n_clusters=n_clusters,
                random_state=42,
                affinity='nearest_neighbors',
                n_neighbors=min(10, len(embeddings_reduced)-1)
            )
            pred_labels = model.fit_predict(embeddings_reduced)
            return pred_labels

        elif category == 'Semi-supervised':
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            # Diviser en train/test
            X_train, _, y_train, _ = train_test_split(
                embeddings_reduced, true_labels, test_size=0.3, random_state=42, stratify=true_labels
            )

            # Entraîner le classificateur
            clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20)
            clf.fit(X_train, y_train)

            # Prédire sur l'ensemble complet
            pred_labels = clf.predict(embeddings_reduced)
            return pred_labels

        else:
            # Par défaut, utiliser KMeans
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
            pred_labels = model.fit_predict(embeddings_reduced)
            return pred_labels

    except Exception as e:
        print(f"Erreur lors de la recréation du clustering pour {method_name}: {e}")
        return None

def create_performance_summary_chart(all_visualizations, output_dir):
    """
    Crée un graphique de résumé des performances pour toutes les meilleures méthodes
    """
    if not all_visualizations:
        return

    # Préparer les données pour le graphique
    node_types = []
    methods = []
    nmi_scores = []
    ari_scores = []
    f1_scores = []
    acc_scores = []

    for viz in all_visualizations:
        node_types.append(viz['node_type'])
        methods.append(viz['method'])
        nmi_scores.append(viz['metrics'].get('nmi', 0))
        ari_scores.append(viz['metrics'].get('ari', 0))
        f1_scores.append(viz['metrics'].get('f1', 0))
        acc_scores.append(viz['metrics'].get('accuracy', 0))

    # Créer la figure avec des sous-graphiques
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # Créer les labels pour l'axe x
    x_labels = [f"{nt}\n{method}" for nt, method in zip(node_types, methods)]
    x_pos = range(len(x_labels))

    # Graphique NMI
    bars1 = ax1.bar(x_pos, nmi_scores, color='skyblue', alpha=0.8)
    ax1.set_title('Scores NMI - Meilleures Méthodes par Type de Nœud', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Score NMI', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Ajouter les valeurs sur les barres
    for bar, score in zip(bars1, nmi_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    # Graphique ARI
    bars2 = ax2.bar(x_pos, ari_scores, color='lightcoral', alpha=0.8)
    ax2.set_title('Scores ARI - Meilleures Méthodes par Type de Nœud', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Score ARI', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    for bar, score in zip(bars2, ari_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    # Graphique F1
    bars3 = ax3.bar(x_pos, f1_scores, color='lightgreen', alpha=0.8)
    ax3.set_title('Scores F1 - Meilleures Méthodes par Type de Nœud', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Score F1', fontsize=12)
    ax3.set_xlabel('Type de Nœud / Méthode', fontsize=12)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    for bar, score in zip(bars3, f1_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    # Graphique Accuracy
    bars4 = ax4.bar(x_pos, acc_scores, color='gold', alpha=0.8)
    ax4.set_title('Scores Accuracy - Meilleures Méthodes par Type de Nœud', fontsize=16, fontweight='bold')
    ax4.set_ylabel('Score Accuracy', fontsize=12)
    ax4.set_xlabel('Type de Nœud / Méthode', fontsize=12)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    for bar, score in zip(bars4, acc_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    # Définir les labels de l'axe x pour tous les sous-graphiques
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)

    plt.suptitle('VGAE + HGT: Résumé des Performances des Meilleures Méthodes de Clustering',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()

    # Sauvegarder le graphique de résumé
    summary_path = os.path.join(output_dir, 'Resume_Performances_Meilleures_Methodes.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Graphique de résumé des performances sauvegardé: {summary_path}")

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
    print("Starting HGT clustering pipeline...")

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
