Proposed a ml stacking algorithm to predict the liver disease. By using the Python tools.
The stacking of all the classification algorithms as a Base learner and  with a meta learner as LR gave a accuracy of 95%.
The preprocessing techniques used helped a lot to acheive this.
## 1.1 Data Pre-Processing
The data preprocessing pipeline prepares the clinical dataset for reliable model learning. It includes categorical encoding, duplicate removal, imputation, skewness transformation, data balancing and feature scaling.

### 1.1.1 Categorical Data Encoding
The dataset contains one categorical feature — **Gender**.  
To make the dataset compatible with machine learning models, the classes are encoded as:
- Female → 0
- Male → 1

### 1.1.2 Duplicate Data Detection
Duplicates introduce redundancy and reduce system efficiency.  
The dataset initially contained **13 duplicate records**, which were detected and removed using Python functions.

### 1.1.3 Imputation
The dataset contains missing values. The **Iterative Imputer** from scikit-learn, powered by **Bayesian Ridge Regression**, is used to predict and fill missing values based on correlations among other attributes.

### 1.1.4 Data Transformation
Several biochemical features exhibited skewed distributions, which influence feature importance and model interpretation.  
The following features were transformed using **log1p transformation**:

- Albumin/Globulin Ratio  
- Total Bilirubin  
- Alkaline Phosphatase  
- Aspartate Aminotransferase  
- Direct Bilirubin  
- Alanine Aminotransferase  

Transformation formula:  
\[
D_\text{new} = \log(1 + D)
\]

### 1.1.5 Data Sampling
The dataset was **highly imbalanced** with:
- 416 patients with liver disease
- 167 patients without liver disease

To mitigate bias, the minority class (no disease) was **upsampled to 416 records** using resampling, producing a balanced dataset for model training.

---

## 1.2 Scaling Features
Feature scaling ensures fair contribution of all variables by eliminating magnitude-based dominance. Multiple scaling techniques were evaluated:

| Technique | Range | Formula |
|----------|--------|----------|
| Min-Max Normalization | 0 to 1 | \((D - D_{min}) / (D_{max} - D_{min})\) |
| Max-Absolute Scaling | −1 to +1 | \(D / D_{maxabs}\) |
| Standardization (Z-Score) | Mean 0, Std 1 | \((D - D_{mean}) / \sigma\) |
| Robust Scaling | Handles outliers strongly | \((D - D_{median}) / IQR\) |

---

## 1.3 Feature Extraction / Selection
Feature selection isolates attributes that most significantly influence the target variable, enhancing model performance and reducing complexity.

### Univariate Feature Selection
Statistical tests performed using scikit-learn:
- Chi-Squared Test
- F-Test (One-Way ANOVA)
- Mutual Info Classifier

### Feature Importance Ranking
Feature importance scores were derived from tree-based models:
- Extra Trees Classifier
- Random Forest Classifier
- LightGBM Classifier

### Correlation Matrix
Pearson’s correlation coefficient was computed to identify relationships among features.  
Features strongly correlated to the target were retained, while redundant feature pairs were pruned.

---

## 1.4 Model Training
A variety of machine learning models were trained and evaluated, with emphasis on ensemble-based methods to increase accuracy and robustness.

### Gradient Boosting Classifier
Builds weak learners iteratively, with every new model correcting the mistakes of the previous model until optimal convergence.

### XGBoost
Optimized and highly scalable version of Gradient Boosting featuring tree pruning, regularization, and parallel execution.

### Bagging
Reduces prediction variance by training multiple base estimators on random subsets of the dataset and averaging their predictions.

### Random Forest Classifier
A bagging-based ensemble of decision trees known for its ability to capture complex decision boundaries while limiting overfitting.

### Extra Trees Classifier
A faster and more randomized version of Random Forest where split thresholds are selected randomly, improving training efficiency.

### Ensemble Stacking Classifier
Multiple base models (**Extra Trees, Random Forest, XGBoost**) are trained first.  
Their predictions are then combined by a **meta-learner (Logistic Regression)** to generate the final output.

---

## Performance Measures
Models were evaluated using standard classification metrics:

| Metric | Formula |
|--------|----------|
| Accuracy | \((TP + TN) / (TP + TN + FP + FN)\) |
| Precision | \(TP / (TP + FP)\) |
| Recall | \(TP / (TP + FN)\) |
| F1-Score | \(2 × (P × R) / (P + R)\) |

---

## 1.5 Feature Visualization
High-dimensional feature interactions were visualized using:

- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- **UMAP (Uniform Manifold Approximation and Projection)**

Both methods reduce the feature space to **2D**, enabling intuitive visualization of the separation between liver disease and non-liver disease groups.

