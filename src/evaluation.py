# This file will handle formatting experiment results & saving results

import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_validate


def evaluate_model(model_name, strategy, pipeline, X, y, cv):
    
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'pr_auc': 'average_precision'
        }
    )
    
    result = {
    "Model": model_name,
    "Strategy": strategy,
    "Accuracy": round(scores['test_accuracy'].mean(), 4),
    "Precision": round(scores['test_precision'].mean(), 4),
    "Recall": round(scores['test_recall'].mean(), 4),
    "F1": round(scores['test_f1'].mean(), 4),
    "ROC-AUC": round(scores['test_roc_auc'].mean(), 4),
    "PR-AUC": round(scores['test_pr_auc'].mean(), 4)
    }
    
    return result

def save_results(new_results):
    """
    Save results to CSV safely:
    - Accepts DataFrame or dict
    - Avoids duplicates
    """
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent 
    results_path = PROJECT_ROOT / "results" / "experiment_results.csv"

    # Convert dict → DataFrame
    if isinstance(new_results, dict):
        new_df = pd.DataFrame([new_results])
    else:
        new_df = new_results.copy()
    
    # If file exists → merge & remove duplicates
    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        
        combined = pd.concat([existing_df, new_df])
        
        # Remove duplicates based on Model + Strategy
        combined = combined.drop_duplicates(subset=["Model", "Strategy"])
        
        combined.to_csv(results_path, index=False)
    
    else:
        new_df.to_csv(results_path, index=False)