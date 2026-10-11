# Employee Attrition Prediction with an Artificial Neural Network

## Why this project matters

Employee attrition can increase recruitment costs, reduce productivity, disrupt teams and result in the loss of organisational knowledge. This project explores whether employee data can reveal useful patterns associated with attrition and support earlier, human-led retention activity.

The aim is not simply to achieve a high accuracy score. The analysis considers class imbalance, the cost of missed leavers, target leakage and the ethical limits of using employee data for prediction.

## What the data revealed

The dataset contains 10,000 synthetic employee records and 29 variables covering tenure, salary, satisfaction, work-life balance, development indicators and employment characteristics.

- 18.67% of employees left, creating a clear class imbalance.
- **Time since last promotion** showed the strongest relationship with leaving (correlation 0.48). Employees promoted within the last three years left at 5–18%, compared with 57% at four years and 72% at five years without a promotion.
- Employees not considered for future training left at 24.5%, compared with 13.2% of those who were.
- Weaker work-life balance (correlation -0.22) and lower satisfaction (-0.14) were more common among leavers.
- Salary showed almost no relationship with leaving (correlation 0.01), suggesting development and recognition mattered more than pay in this dataset.
- These relationships are associations and should not be interpreted as evidence of causation.

## Approach

The project follows four stages:

1. Inspect the data and assess missing values, class balance and feature distributions.
2. Remove identifiers, administrative fields and post-outcome information (such as leaving date and employment status) that could introduce leakage.
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

## Ethical considerations

- **Protected characteristics:** gender and age are included as model inputs. In this dataset, women left at a higher rate (22.8%) than men (15.2%), so the model may learn to treat gender itself as a risk factor. Any real use would require retraining without protected characteristics and testing that predictions are fair across groups.
- **Possible leakage:** time since promotion and the manager-assessed retention risk may be recorded close to, or influenced by, an employee's decision to leave. Their strength should be validated with data captured well before any leaving date.
- **Human-led use only:** a prediction should prompt a supportive conversation (development, workload, recognition), never an automated decision about an individual.

## Limitations and next steps

- Compare the ANN with logistic regression and tree-based baseline models.
- Use a stratified split and cross-validation for more robust evaluation.
- Test class weighting, resampling and decision-threshold tuning.
- Add validation monitoring and early stopping to reduce overfitting risk.
- Report F1, PR-AUC and ROC-AUC alongside accuracy, precision and recall.
- Build preprocessing and inference into one reproducible pipeline.
- Retrain without protected characteristics and evaluate subgroup performance and fairness before any operational use.

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

## Data note

The dataset is synthetic and created for educational purposes. Names, employee IDs and birth dates have been replaced with neutral values.

## Tools and technologies

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · TensorFlow/Keras · Jupyter Notebook
