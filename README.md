# Deep-Graph-Clustering-in-the-context-of-Linked-Data


This repository contains the code and resources for my Master's thesis project, focused on clustering algorithms and their application to heterogeneous bibliographic graphs (DBLP dataset).

## Project Structure

- **Global fin/**: Scripts and notebooks for global clustering (all node types together).
- **par type/**: Scripts and notebooks for clustering by node type (author, publication, venue, etc.).

## Main Features

- **Extraction and processing of heterogeneous graphs** from DBLP.
- **Learning embeddings** with VGAE (Variational Graph AutoEncoder) and HGT (Heterogeneous Graph Transformer).
- **Advanced clustering**: K-Means, Agglomerative (hierarchical), Spectral Clustering, DEC (Deep Embedded Clustering), GMM, semi-supervised methods.
- **Evaluation**: NMI, ARI, F1-score, Accuracy, Purity.
- **Visualization**: t-SNE, confusion matrices, cluster visualization by type and globally.


## Requirements

- Python 3.8+
- PyTorch, DGL, scikit-learn, matplotlib, pandas, rdflib, sentence-transformers, umap-learn, etc.


## Usage

1. **Clone the repository**  
   ```sh
   git clone <repository_url>
   cd "les codes clustering"
   ```

2. **Install dependencies**  
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the notebooks**  
   - For global clustering:  
     Open and run [`Global fin/vgae-globale.py`](Global%20fin/vgae-globale.py) or [`Global fin/hgt_dec.ipynb`](Global%20fin/hgt_dec.ipynb)
   - For clustering by type:  
     Open and run [`par type/vgae.py`](par%20type/vgae.py) or [`par type/kmean.ipynb`](par%20type/kmean.ipynb)


## License

This project is distributed under the MIT License.

---

*For any questions or suggestions, feel free to contact me!*
