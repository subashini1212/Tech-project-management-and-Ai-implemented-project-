from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "delay_model.pkl"


def make_demo_dataset():
    """
    Creates illustrative training data for learning/demo purposes.
    Replace this with real, correctly labelled historical project data
    before using predictions for real decisions.
    """
    rows = []
    for total in range(5, 51, 3):
        for completed_pct in (0.2, 0.4, 0.6, 0.8):
            for days in (2, 5, 10, 20):
                for team in (1, 2, 4, 6):
                    for avg_delay in (0, 1, 3, 5):
                        completed = int(total * completed_pct)
                        remaining = total - completed

                        # Demonstration label rule used to create synthetic examples.
                        pressure = (remaining / max(team, 1)) + avg_delay * 2
                        risk = int(days <= 5 and pressure > 8)
                        rows.append({
                            "total_tasks": total,
                            "completed_tasks": completed,
                            "days_remaining": days,
                            "team_size": team,
                            "avg_task_delay": avg_delay,
                            "delay_risk": risk
                        })
    return pd.DataFrame(rows)


def train_and_save_model():
    data = make_demo_dataset()
    feature_columns = [
        "total_tasks", "completed_tasks", "days_remaining",
        "team_size", "avg_task_delay"
    ]
    X = data[feature_columns]
    y = data["delay_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Demo test accuracy:", round(accuracy_score(y_test, predictions), 3))
    print(classification_report(y_test, predictions, zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    train_and_save_model()
