# MolMutNet: GNN-Based Mutagenicity Prediction

MolMutNet is a deep learning project that leverages the power of Graph Neural Networks (GNNs) to classify chemical compounds as mutagenic or non-mutagenic. [cite_start]By representing molecules as graphs—where atoms are nodes and bonds are edges—this project demonstrates an effective method for predicting molecular properties directly from their structure[cite: 1, 2].

<p align="center">
  <img src="https://i.imgur.com/your_image_link_for_non_mutagenic.png" alt="Non-Mutagenic Molecule" width="45%">
  &nbsp; &nbsp;
  <img src="https://i.imgur.com/your_image_link_for_mutagenic.png" alt="Mutagenic Molecule" width="45%">
</p>
<p align="center">
  <em>Figure: Sample non-mutagenic (left) and mutagenic (right) molecules from the MUTAG dataset.</em>
</p>

---

### 💡 Overview

Predicting the mutagenicity of a compound is a critical step in drug discovery and chemical safety assessment, helping to identify potentially harmful substances early on. This project utilizes GNNs, a class of neural networks designed for graph-structured data, to learn the intricate topological features of molecules and make accurate predictions. [cite_start]We implement, tune, and compare two popular GNN architectures: a **Graph Convolutional Network (GCN)** and **GraphSAGE**[cite: 1, 2].

---

### ✨ Key Features

* [cite_start]**Graph-Based Molecular Analysis**: Treats molecules as graphs to capture rich structural information, with atoms as nodes and bonds as edges[cite: 1, 2].
* [cite_start]**Two GNN Architectures**: Implements and compares both GCN and GraphSAGE models for the classification task[cite: 1, 2].
* [cite_start]**Hyperparameter Tuning**: Optimizes model performance by conducting a grid search over parameters like hidden channels, dropout rate, learning rate, and weight decay[cite: 2].
* [cite_start]**Detailed Performance Evaluation**: Provides comprehensive evaluation metrics, including accuracy, precision, recall, F1-score, and confusion matrices[cite: 2].
* [cite_start]**Data Visualization**: Includes tools to visualize the molecular graphs from the dataset, labeling atoms and bond types for clear interpretability[cite: 1, 2].

---

### 🔬 Dataset: The MUTAG Collection

This project uses the **MUTAG** dataset, a standard benchmark for graph classification tasks. [cite_start]It consists of 188 chemical compounds labeled for their mutagenic effect on a bacterium[cite: 1].

**Dataset Statistics:**
* [cite_start]**Total Graphs**: 188 molecules [cite: 1]
* [cite_start]**Number of Classes**: 2 (Non-Mutagenic / Mutagenic) [cite: 1]
* [cite_start]**Node Features**: 7, representing one-hot encoded atom types (C, N, O, F, I, Cl, Br)[cite: 1, 2].
* [cite_start]**Edge Features**: 4, representing one-hot encoded bond types (Aromatic, Single, Double, Triple)[cite: 1, 2].

---

### ⚙️ Modeling Approach

The project follows a standard workflow for graph-based machine learning:

1.  **Data Loading & Preprocessing**: The MUTAG dataset is loaded using `torch_geometric`. [cite_start]Node features are normalized to stabilize training[cite: 2].
2.  [cite_start]**Data Splitting**: The dataset is split into an 80% training set and a 20% testing set[cite: 2].
3.  **Model Architecture**: Both GCN and GraphSAGE models are built with the following structure:
    * [cite_start]Two graph convolution layers with **Leaky ReLU** activation[cite: 2].
    * [cite_start]**Batch Normalization** after each convolutional layer to improve training stability[cite: 2].
    * [cite_start]**Dropout** for regularization to prevent overfitting[cite: 2].
    * [cite_start]A **Global Mean Pooling** layer to aggregate node features into a single graph-level representation[cite: 1, 2].
    * [cite_start]A final **Linear Layer** with a Log-Softmax activation for classification[cite: 1, 2].
4.  **Training & Optimization**: The models are trained using:
    * [cite_start]**Loss Function**: Cross-Entropy Loss, suitable for multi-class classification tasks[cite: 2].
    * [cite_start]**Optimizer**: The Adam optimizer, which adaptively adjusts the learning rate[cite: 1, 2].
5.  [cite_start]**Hyperparameter Tuning**: A grid search is performed to find the optimal combination of hidden channels, dropout rate, learning rate, and weight decay, maximizing the model's performance on the test set[cite: 2].

---

### 📊 Performance & Results

[cite_start]After hyperparameter tuning, both the GCN and GraphSAGE models achieved a **peak accuracy of 89.47%** on the test set[cite: 2]. The detailed performance of the best GCN model is presented below.

#### GCN Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| 0 (Non-Mutagenic) | 0.90 | 0.75 | 0.82 | 12 |
| 1 (Mutagenic) | 0.89 | 0.96 | 0.93 | 26 |
| **Accuracy** | | | **0.89** | **38** |
| **Macro Avg** | **0.90** | **0.86** | **0.87** | **38** |
| **Weighted Avg** | **0.90** | **0.89** | **0.89** | **38** |

[cite_start]*Classification report generated using `scikit-learn` on the test data predictions*[cite: 2].

#### GCN Confusion Matrix
<p align="center">
  <img src="https://i.imgur.com/your_confusion_matrix_image.png" alt="GCN Confusion Matrix" width="50%">
</p>

These results demonstrate the model's strong ability to correctly identify mutagenic compounds, a crucial requirement for a reliable chemical screening tool.

---

### 🚀 Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/meanderinghuman/MolMutNet.git](https://github.com/meanderinghuman/MolMutNet.git)
    cd MolMutNet
    ```

2.  **Install the required dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install torch torch_geometric scikit-learn matplotlib seaborn networkx
    ```

3.  **Run the Jupyter Notebook:**
    Launch Jupyter and open the `ANN Phase 2.ipynb` notebook to see the data loading, model training, and evaluation process.
    ```bash
    jupyter notebook "ANN Phase 2.ipynb"
    ```

---

### 🛠️ Technologies Used

* [cite_start]**PyTorch** [cite: 1, 2]
* [cite_start]**PyTorch Geometric (PyG)** [cite: 1, 2]
* [cite_start]**Scikit-learn** [cite: 1, 2]
* [cite_start]**NetworkX** [cite: 1, 2]
* [cite_start]**Matplotlib & Seaborn** [cite: 1, 2]

---

### 🔮 Future Directions

* **Explore Advanced Architectures**: Implement other GNN layers like Graph Attention Networks (GAT) or Graph Isomorphism Networks (GIN) to potentially improve accuracy.
* **Larger Datasets**: Test the models on more extensive molecular datasets like Tox21 to assess scalability and generalization.
* **Explainability**: Incorporate GNN explainability techniques (e.g., GNNExplainer) to understand which molecular substructures are most influential in predicting mutagenicity.
* **Web Application**: Deploy the trained model as a web service or API where users can input a molecule (e.g., via SMILES string) and get a prediction.

---

### 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

<details>
<summary>MIT License Text</summary>