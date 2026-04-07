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
            'pr_auc': 'average_precision',
            'roc_auc': 'roc_auc'
        }
    )

    result = {
        "Model": model_name,
        "Strategy": strategy,
        "Accuracy": f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
        "Precision": f"{scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}",
        "Recall": f"{scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}",
        "F1": f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
        "PR-AUC": f"{scores['test_pr_auc'].mean():.4f} ± {scores['test_pr_auc'].std():.4f}",
        "ROC-AUC": f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}"
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