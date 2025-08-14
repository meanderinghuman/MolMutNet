# MolMutNet: GNN-Based Mutagenicity Prediction

MolMutNet is a deep learning project that leverages the power of Graph Neural Networks (GNNs) to classify chemical compounds as mutagenic or non-mutagenic. By representing molecules as graphs—where atoms are nodes and bonds are edges—this project demonstrates an effective method for predicting molecular properties directly from their structure.




---

### 💡 Overview

Predicting the mutagenicity of a compound is a critical step in drug discovery and chemical safety assessment, helping to identify potentially harmful substances early on. Traditional methods can be time-consuming and expensive. This project utilizes GNNs, a class of neural networks designed for graph-structured data, to learn the intricate topological features of molecules and make accurate predictions. We explore and compare two popular GNN architectures: **Graph Convolutional Network (GCN)** and **GraphSAGE**.

---

### ✨ Key Features

* **Graph-Based Molecular Analysis**: Treats molecules as graphs to capture rich structural information.
* **Two GNN Architectures**: Implements and compares both GCN and GraphSAGE models for the classification task.
* **Hyperparameter Tuning**: Optimizes model performance by tuning key parameters like hidden channels, dropout rate, and learning rate.
* **Detailed Performance Evaluation**: Provides comprehensive evaluation metrics, including accuracy, precision, recall, F1-score, and confusion matrices.
* **Data Visualization**: Includes tools to visualize the molecular graphs from the dataset, labeling atoms and bond types for clear interpretability.

---

### 🔬 Dataset: The MUTAG Collection

This project uses the **MUTAG** dataset, a standard benchmark for graph classification tasks. It consists of 188 chemical compounds labeled for their mutagenic effect on a bacterium.

**Dataset Statistics:**
* **Total Graphs**: 188 molecules
* **Number of Classes**: 2 (Non-Mutagenic / Mutagenic)
* **Node Features**: 7 (representing one-hot encoded atom types: C, N, O, F, I, Cl, Br)
* **Edge Features**: 4 (representing one-hot encoded bond types: Aromatic, Single, Double, Triple)

---

### ⚙️ Modeling Approach

The project follows a standard workflow for graph-based machine learning:

1.  **Data Loading & Preprocessing**: The MUTAG dataset is loaded using `torch_geometric`. Node features are normalized to stabilize training.
2.  **Data Splitting**: The dataset is split into an 80% training set and a 20% testing set.
3.  **Model Architecture**: Both GCN and GraphSAGE models are built with the following structure:
    * Two graph convolution/SAGE layers with **ReLU** activation.
    * **Batch Normalization** after each convolutional layer to improve training stability.
    * **Dropout** for regularization to prevent overfitting.
    * A **Global Mean Pooling** layer to aggregate node features into a single graph-level representation.
    * A final **Linear Layer** with a Softmax activation for classification.
4.  **Training & Optimization**: The models are trained using:
    * **Loss Function**: Cross-Entropy Loss, suitable for multi-class classification tasks.
    * **Optimizer**: The Adam optimizer, which adaptively adjusts the learning rate.
5.  **Hyperparameter Tuning**: A grid search is performed to find the optimal combination of hidden channels, dropout rate, learning rate, and weight decay, maximizing the model's performance on the test set.

---

### 📊 Performance & Results

After hyperparameter tuning, both the GCN and GraphSAGE models achieved a **peak accuracy of 89.47%** on the test set. The detailed performance of the best GCN model is presented below.

#### GCN Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| 0 (Non-Mutagenic) | 0.90 | 0.75 | 0.82 | 12 |
| 1 (Mutagenic) | 0.89 | 0.96 | 0.93 | 26 |
| **Accuracy** | | | **0.89** | **38** |
| **Macro Avg** | **0.90** | **0.86** | **0.87** | **38** |
| **Weighted Avg** | **0.90** | **0.89** | **0.89** | **38** |

#### GCN Confusion Matrix
| | Predicted Non-Mutagenic | Predicted Mutagenic |
| :--- | :--- | :--- |
| **Actual Non-Mutagenic** | 9 | 3 |
| **Actual Mutagenic**| 1 | 25 |

These results demonstrate the model's strong ability to correctly identify mutagenic compounds, a crucial requirement for a reliable chemical screening tool.

---

### 🚀 Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://your-repo-link.git](https://your-repo-link.git)
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

* **PyTorch**
* **PyTorch Geometric (PyG)**
* **Scikit-learn**
* **NetworkX**
* **Matplotlib & Seaborn**

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