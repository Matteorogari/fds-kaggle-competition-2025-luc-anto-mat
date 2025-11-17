# Super-Team of Trainers: Ensemble Stacking for Pokémon Battle Prediction
<div align="center">
   <img src="./assets/team.jpg" width="55%"/>
</div>
This repository contains the code and methodology for the project:  **Super-Team of Trainers: Modeling Complex Outcomes with an Ensemble Stacking Approach to Pokémon Battle Prediction**  .

The project addresses the task of predicting simulated Pokémon battle outcomes by modeling the problem as a **binary classification** task to estimate Player 1's victory probability.

---

## 📄 Abstract

This report analyzes the machine learning methodologies developed for the Pokémon Battle Prediction competition. The problem is modeled as a binary classification task to estimate Player 1's victory probability, based on three systematically varied sets of input features. The core approach across all implementations is a **Two-Level Stacking Ensemble**, which leverages the complementarity between linear predictors (Logistic Regression) and boosting models (XGBoost, HGBT). The results strongly demonstrate that integrating **domain-specific metrics**, specifically those that quantify strategic type advantage and residual in-battle power, is essential for maximizing predictive reliability.

---
<div align="center">
   <img src="./assets/Pirate.jpg" width="35%"/>
</div>

## 🧠 Modeling Architecture: Stacking Ensemble

The Stacking Ensemble architecture was chosen to combine the **stability of linear models** with the **non-linearity resolving power of tree-based models**. 

### Tier-0 Base Learners

The base model set was diversified to ensure the complementarity of Out-Of-Fold (OOF) predictions, including:
* **Logistic Regression (LR):** For stability and low-bias.
* **Boosting Models (XGBoost, HGBT):** For capturing non-linear interactions.

### Tier-1 Meta-Model Weights (XGBoost)
The Tier-1 **XGBoost Meta-Model** effectively balances these signals, prioritizing linear stability:

| Base Model | Importance (Weight) | Role |
| :--- | :--- | :--- |
| **LR LITE** | **42.81%** | Provides a stable (low-correlation) signal to tree models, stabilizing the ensemble. |
| XGBoost / HGBT / RF | 20.34% | Adds the capacity to correct complex classification errors that LR, by its nature, cannot resolve. |

<div align="center">
  <img src="./assets/pika.png" width="35%">
</div>
---

## ⚙️ Feature Engineering: Domain-Specific Metrics

The analysis validates the hypothesis that **complex strategic features provide incremental predictive value**.

### Strategic Features (17 Set)

The performance boost was driven by these domain-specific metrics:
* **Meta Type Advantage Metric:** Calculates cross-damage multipliers. *Rationale: The type system is the most critical matchup mechanic*. 
* **Weighted Active Leader Power:** An index that weights the active Pokémon's stats, with higher importance assigned to **Speed (Spe)**, reflecting the strategic value of initiative.
* **Attrition & Condition:** Includes features like `Delta Mean HP` and `Delta Status/Effect Score`.

---

## 🛠️ Repository Structure

* **`.github/workflows/`**: GitHub Actions workflow for automated LaTeX compilation.
* **`sec/`**: Individual sections of the LaTeX document.
* **`src/`**: Source code and modeling scripts (e.g., `main17.py`).
* **`main.tex`**: The main LaTeX document file.
<div align="center">
  <img src="./assets/comp.png" width="55%"/>
</div>
