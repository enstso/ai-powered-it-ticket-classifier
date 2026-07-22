import pandas as pd
from pathlib import Path
from train import model_a_pipeline, model_b_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data/clean_tickets.csv"

    df = pd.read_csv(csv_path)

    X = df["text"]
    y = df["category"]

    model_a = model_a_pipeline()
    model_b = model_b_pipeline()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    accuracy_scores_a = cross_val_score(
        model_a,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    accuracy_scores_b = cross_val_score(
        model_b,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print(f"model a (simple processing) accuracy {accuracy_scores_a.mean()} +/- {accuracy_scores_a.std()}")
    print(f"model b (ntlk processing) accuracy {accuracy_scores_b.mean()} +/- {accuracy_scores_b.std()}")

    """
    model a (simple processing) accuracy 0.7924242424242424 +/- 0.0869860611782262
    model b (ntlk processing) accuracy 0.7257575757575758 +/- 0.056772709076349105
    """

    precision_scores_a = cross_val_score(
        model_a,
        X,
        y,
        cv=cv,
        scoring="precision_macro"
    )

    precision_scores_b = cross_val_score(
        model_b,
        X,
        y,
        cv=cv,
        scoring="precision_macro"
    )

    print(f"model a (simple processing) precision {precision_scores_a.mean()} +/- {precision_scores_a.std()}")
    print(f"model b (ntlk processing) precision {precision_scores_b.mean()} +/- {precision_scores_b.std()}")

    """
       model a (simple processing) precision 0.7966666666666666 +/- 0.10653637876331258
       model b (ntlk processing) precision 0.705 +/- 0.08255469567370334   
    """
    recall_scores_a = cross_val_score(
        model_a,
        X,
        y,
        cv=cv,
        scoring="recall_macro"
    )

    recall_scores_b = cross_val_score(
        model_b,
        X,
        y,
        cv=cv,
        scoring="recall_macro"

    )

    print(f"model a (simple processing) recall {recall_scores_a.mean()} +/- {precision_scores_a.std()}")
    print(f"model b (ntlk processing) recall {recall_scores_b.mean()} +/- {precision_scores_b.std()}")

    """
        model a (simple processing) recall 0.7791666666666667 +/- 0.10653637876331258
        model b (ntlk processing) recall 0.7166666666666666 +/- 0.08255469567370334
    """

    f1_scores_a = cross_val_score(
        model_a,
        X,
        y,
        cv=cv,
        scoring="f1_macro"
    )

    f2_scores_b = cross_val_score(
        model_b,
        X,
        y,
        cv=cv,
        scoring="f1_macro"
    )

    print(f"model a (simple processig) f1-score {f1_scores_a.mean()} +/- {precision_scores_a.std()}")
    print(f"model b (ntlk processing) f1-score {f2_scores_b.mean()} +/- {precision_scores_b.std()}")

    """
    model a (simple processig) f1-score 0.7651190476190477 +/- 0.10653637876331258
    model b (ntlk processing) f1-score 0.6938095238095239 +/- 0.08255469567370334
    """


    """
    model a is the best model
    """