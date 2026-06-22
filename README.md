# FYT - EPL Direct Free-Kick Goal Decline Framework

This repository contains the source code, predictive machine learning models, datasets, and tactical simulations for the university dissertation:
**"Design of Data Driven Framework to analyze the decline of Free-Kick goals in Premier League: Machine Learning and Tactical Analysis"**

## 🏟️ Core Features

* **Tactical Free-Kick Simulator**: Interactive top-down pitch and 2D goalmouth view representing projected defensive walls, crocodile defenders, goalkeeper reach, and curling trajectories.
* **Pre-Shot xG vs. Post-Shot xG (PSxG) Models**: An XGBoost classifier trained on historical EPL direct free-kick shots to compute pre-shot and post-shot scoring probabilities.
* **Dissertation Insights & Analytics**: Live 10-season decline trends, spatial density comparisons, and statistical hypothesis testing (Welch's T-Test and Chi-Square tests).
* **Explainable AI (SHAP)**: Global and local explainability models displaying tactical parameter contributions.
* **Elite Dynamic Styling**: Three custom-built visual style themes (Opta Vision Midnight, StatsBomb Forest, and Oxford Academic Light).

## 🚀 Setup & Execution

### 1. Installation
Ensure Python 3.9+ is installed, then install the required dependencies:
```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn shap joblib scipy mplsoccer==1.5.1 streamlit
```

### 2. Launch the Application
Run the Streamlit server from the project directory:
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.
