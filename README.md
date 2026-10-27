# Bankruptcy Prevention Classifier Model

An end-to-end machine learning classification system designed to analyze corporate financial risk parameters and predict bankruptcy vulnerability. This project processes historical risk indicators to build predictive models and deploys them via an interactive user dashboard for real-time inference.

## 📊 Core Project Overview
* **Objective:** Build a multi-class predictive engine capable of classifying corporate financial states into distinct risk profiles based on operational indices.
* **Modeling Pipeline:** Explores time-series analytics, feature distributions, variance evaluation, and model benchmarking steps.
* **Production Stack:** Developed natively in Python, utilizing standard classification frameworks, and serialized for deployment within a lightweight dashboard interface.

## 📂 Repository Architecture
* `Bankrupt_prevention.ipynb` — Full data cleaning, exploratory diagnostics, model training iterations, and performance matrices.
* `app (1).py` — Streamlit production script managing the web-based interactive operator workspace.
* `requirements.txt` — Unified dependency configurations managing production package parameters (`streamlit`, `scikit-learn`, `pandas`, `numpy`).
* `Bankruptcy_Prevention_Presentation presentation.pptx` — Polished slide deck mapping project context, engineering metrics, and strategic outcomes.

## 🛠️ Execution & Deployment
1. Install localized dependencies tracked in the configurations pipeline:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the interactive operator production workspace interface:
   ```bash
   streamlit run "app (1).py"
   ```# Bankruptcy-Prevention-Pipeline
