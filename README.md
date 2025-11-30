# Credit Card Approval Prediction

This repository implements a full end-to-end machine learning pipeline to predict **creditworthiness** using demographic features and credit behaviour data from the Kaggle Credit Approval Dataset.  
Because the raw dataset does **not** provide an explicit approval/decline label, we construct a behaviour-driven target from delinquency history and evaluate several model classes under severe class imbalance.

Our goal is to compare **linear**, **tree-based**, and **neural network** approaches and identify which modelling family best detects high-risk clients while minimising costly false negatives.

---

## 📁 Repository Layout

project/
│
├── README.md                  # You are here
├── requirements.txt           # Python dependencies
│
├── data/                      # Raw + cleaned + processed datasets
│   ├── application_record.csv
│   ├── credit_record.csv
│   ├── clean_merged.csv
│   ├── X_train.csv, y_train.csv
│   ├── X_test.csv, y_test.csv
│   ├── X_train_processed.csv, X_test_processed.csv
│   ├── X_train_smote.csv, y_train_smote.csv
│   └── …
│
├── src/                       # Notebooks for each pipeline stage
│   ├── data_cleaning.ipynb
│   ├── data_processing.ipynb
│   ├── logistic_regression.ipynb
│   ├── random_forest.ipynb
│   ├── Random_forest_Vivian.ipynb
│   ├── final_random_forest.ipynb
│   └── neural_network.ipynb
│
└── venv/                      # Optional local virtual environment

---

## 📊 Data Files (`data/`)

- **application_record.csv** — Raw applicant information.
- **credit_record.csv** — Monthly credit behaviour.
- **clean_merged.csv** — Final cleaned dataset with engineered features + behaviour-derived target.
- **Train/test splits** — Stratified 80/20 split.
- **Processed features** — After scaling + encoding.
- **SMOTE-balanced data** — For models requiring balanced inputs.

---

## 📒 Notebooks (`src/`)

### 1. data_cleaning.ipynb
- Cleans raw files, resolves duplicates, constructs engineered features and behaviour-based labels.

### 2. data_processing.ipynb
- Additional feature engineering, preprocessing, scaling, encoding, SMOTE, and export.

### 3. logistic_regression.ipynb
- Linear baseline, class weighting vs SMOTE, CV hyperparameter tuning.

### 4. Random Forest notebooks
- Baseline RF, alternative versions, final tuned RF with engineered features.

### 5. neural_network.ipynb
- Feedforward MLP with BatchNorm, Dropout, SMOTE-based training, scheduler, and ROC-AUC optimisation.

---

## 🔁 Typical Workflow

1. Create environment (optional)
   python -m venv venv  
   source venv/bin/activate  
   venv\Scripts\activate (Windows)

2. Install dependencies  
   pip install -r requirements.txt

3. Run notebooks in order:  
   data_cleaning → data_processing → the modelling notebooks.

4. Ensure working directory is project root for correct relative paths.

---

## 📝 Notes

- Kaggle raw data is large; keep out of version control.
- Add a models/ folder if saving model artefacts.
- Severe class imbalance (~98% good clients) requires cost-sensitive evaluation and threshold tuning.