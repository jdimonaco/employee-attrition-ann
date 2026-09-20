# Employee Attrition Prediction with an Artificial Neural Network

## Why this project matters

Employee attrition can increase recruitment costs, reduce productivity, disrupt teams and result in the loss of organisational knowledge. This project explores whether employee data can reveal useful patterns associated with attrition and support earlier, human-led retention activity.

The aim is not simply to achieve a high accuracy score. The analysis considers class imbalance, the cost of missed leavers, target leakage and the ethical limits of using employee data for prediction.

## What the data revealed

The dataset contains 10,000 employee records and 29 variables covering tenure, salary, satisfaction, work-life balance, development indicators and employment characteristics.

- 18.67% of employees left, creating a clear class imbalance.
- Shorter tenure, lower satisfaction and weaker work-life balance were more common among leavers.
- Several development- and retention-related variables showed different distributions between employees who stayed and those who left.
- These relationships are associations and should not be interpreted as evidence of causation.

## Approach

The project follows four stages:

1. Inspect the data and assess missing values, class balance and feature distributions.
2. Remove identifiers, administrative fields and post-outcome information that could introduce leakage.
3. Encode categorical variables, scale the features and create an 80/20 train-test split.
4. Train and evaluate a compact feed-forward Artificial Neural Network using TensorFlow/Keras.

The network contains two hidden layers with six ReLU units each and a sigmoid output layer for binary classification.

## Results

The model was evaluated on 2,000 held-out employee records.

| Metric | Result |
| --- | ---: |
| Accuracy | 91.2% |
| Precision for leavers | 83.3% |
| Recall for leavers | 66.5% |
| True negatives | 1,574 |
| False positives | 50 |
| False negatives | 126 |
| True positives | 250 |

The model improved on the 81.2% majority-class accuracy baseline. However, recall is the more important limitation: approximately one-third of employees who left were not identified. The appropriate decision threshold would therefore depend on the practical cost of false positives and false negatives.

## Limitations and next steps

- Compare the ANN with logistic regression and tree-based baseline models.
- Use a stratified split and cross-validation for more robust evaluation.
- Test class weighting, resampling and decision-threshold tuning.
- Add validation monitoring and early stopping to reduce overfitting risk.
- Report F1, PR-AUC and ROC-AUC alongside accuracy, precision and recall.
- Build preprocessing and inference into one reproducible pipeline.
- Evaluate subgroup performance and fairness before any operational use.

This is an analytical proof of concept, not an automated employment decision system. Any retention intervention should remain human-led and consider the wider context behind an individual prediction.

## Repository structure

```text
employee-attrition-ann/
├── data/
│   └── employee_attrition_data.csv
├── employee_attrition_ann.ipynb
├── requirements.txt
└── README.md
```

## Running the project

Clone the repository, install the required packages and open the notebook in Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook employee_attrition_ann.ipynb
```

Run the notebook from top to bottom. The dataset is loaded from the `data` folder using a relative path.

## Tools and technologies

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · TensorFlow/Keras · Jupyter Notebook
