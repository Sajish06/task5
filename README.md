# 🩺 Heart Disease Prediction using Decision Tree and Random Forest

## 📘 Project Overview
This project applies **Decision Tree** and **Random Forest** machine learning algorithms to predict the presence of heart disease based on clinical parameters.  
The implementation and analysis were done using **Google Colab** and **Python (scikit-learn)**.

The work corresponds to **Task 5: Decision Trees and Random Forests**, which aims to understand the working principles of tree-based models, visualize decision boundaries, control overfitting, and compare ensemble performance.

---

## 🎯 Objectives
1. Load and explore the Heart Disease dataset (`heart.csv`).
2. Preprocess the data — handle missing values, encode categorical features, and split into train/test sets.
3. Implement a **Decision Tree Classifier** and analyze overfitting using `max_depth`.
4. Visualize the decision tree using both `plot_tree()` and Graphviz.
5. Train a **Random Forest Classifier** and compare its performance with the Decision Tree.
6. Evaluate models using metrics such as accuracy, confusion matrix, classification report, and cross-validation.
7. Interpret **feature importances** to identify the most influential health indicators.
8. Save the trained model for deployment or reuse.

---

## 📂 Dataset
- **File:** `heart.csv`
- **Source:** Standard Heart Disease dataset (e.g., UCI repository)
- **Target variable:** `target` (1 = heart disease present, 0 = not present)
- **Features include:**
  - `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`,
    `exang`, `oldpeak`, `slope`, `ca`, `thal`

---

## ⚙️ Steps Performed

1. **Environment Setup**
   - Installed dependencies (`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `graphviz`, `pydotplus`).
   - Configured Google Colab for visualization and file I/O.

2. **Data Loading**
   - Loaded `heart.csv` using `pandas`.
   - Verified dataset shape, data types, and column names.

3. **Data Preprocessing**
   - Checked for and filled missing values.
   - Label-encoded categorical columns.
   - Split data into **training (80%)** and **testing (20%)** sets using stratified sampling.

4. **Decision Tree Model**
   - Trained a baseline `DecisionTreeClassifier`.
   - Tuned hyperparameter `max_depth` to control overfitting.
   - Visualized tree structure using `sklearn.tree.plot_tree` and Graphviz (`export_graphviz` + `pydotplus`).
   - Compared training and testing accuracy for different depths.

5. **Random Forest Model**
   - Trained a `RandomForestClassifier` with 100 trees.
   - Compared accuracy and classification report with the Decision Tree.
   - Plotted feature importances to identify key predictors.

6. **Model Evaluation**
   - Calculated accuracy, confusion matrix, and ROC-AUC score.
   - Performed 5-fold **cross-validation** for both models.
   - Compared average CV accuracies and variance.

7. **Model Saving**
   - Saved the best-performing model using `joblib.dump()` as `best_random_forest.pkl`.

8. **Visualization**
   - Plotted:
     - Decision Tree (first few levels)
     - Accuracy vs Depth curve
     - Feature Importance bar chart

---

## 🧰 Libraries Used
| Library | Purpose |
|----------|----------|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `matplotlib` | Visualization |
| `graphviz`, `pydotplus` | Decision tree visualization |
| `joblib` | Model saving and loading |

---

## 🚀 How to Run

### Option 1: Run on Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the notebook file (`Task5_Heart_Disease.ipynb`).
3. Upload the dataset file `heart.csv` to the Colab environment.
4. Run all cells sequentially (`Runtime → Run all`).

### Option 2: Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-decision-tree.git
   cd heart-disease-decision-tree
