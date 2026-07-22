import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.train import model_a_pipeline

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data/clean_tickets.csv"

    df = pd.read_csv(csv_path)

    X = df["text"]
    y = df["category"]

    classes = ["access","network","security","software"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    model = model_a_pipeline()

    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv
    )

    print("=====accuracy score=====\n",accuracy_score(y,y_pred))
    # 0.7931034482758621

    print("=====classification report====\n",classification_report(y,y_pred))
    """
                   precision    recall  f1-score   support

      access       0.92      0.86      0.89        14
     network       0.68      0.88      0.77        17
    security       1.00      0.54      0.70        13
    software       0.75      0.86      0.80        14

    accuracy                           0.79        58
   macro avg       0.84      0.78      0.79        58
weighted avg       0.83      0.79      0.79        58
    """
    print("======confusion_matrix=======\n")
    """
                     Predict : access  Predict : network  Predict : security  Predict : software
Real : access                  12                  1                   0                   1
Real : network                  1                 15                   0                   1
Real : security                 0                  4                   7                   2
Real : software                 0                  2                   0                  12
    """
    matrix = confusion_matrix(
        y,
        y_pred)

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Real : {classe}" for classe in classes],
        columns=[f"Predict : {classe}" for classe in classes]
    )

    print(matrix_df.to_string())


texts = np.asarray(X).astype(str)
expected_categories = np.asarray(y)
predicted_categories = np.asarray(y_pred)

error_mask = expected_categories != predicted_categories

prediction_errors = pd.DataFrame({
    "text":texts[error_mask],
    "expected_category":expected_categories[error_mask],
    "predicted_category": predicted_categories[error_mask]
})

project_root =  Path(__file__).resolve().parent.parent
errors_csv_path = project_root / "reports/prediction_errors.csv"
prediction_errors.to_csv(errors_csv_path,index=False)

print(f"Errors nb : {len(prediction_errors)}")
print(prediction_errors.to_string())