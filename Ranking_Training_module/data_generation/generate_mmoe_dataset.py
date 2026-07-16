# Not to deploy, just to generate data and train or test the neural network.

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.feature_spec import FEATURE_ORDER  # noqa: E402


def generate_synthetic_data(num_users=1000, num_repos=10000, emb_dim=384, output_file="training_data.npz"):
    print(f"Generating {num_users} users and {num_repos} repos...")
    
    # 1. Generate Embeddings directly as float32 to save RAM!
    np.random.seed(42)
    user_embs = np.random.randn(num_users, emb_dim).astype(np.float32)
    user_embs /= np.linalg.norm(user_embs, axis=1, keepdims=True)
    
    repo_embs = np.random.randn(num_repos, emb_dim).astype(np.float32)
    repo_embs /= np.linalg.norm(repo_embs, axis=1, keepdims=True)
    
    # 2. Generate Dense Features for Repos (10 Features total)
    star_counts = np.random.exponential(scale=500, size=num_repos).astype(np.float32)
    fork_counts = (star_counts * np.random.uniform(0.1, 0.4, size=num_repos)).astype(np.float32)
    open_issues = np.random.randint(0, 100, size=num_repos).astype(np.float32)
    pushed_days_ago = np.random.randint(0, 365, size=num_repos).astype(np.float32)
    readme_lengths = np.random.normal(loc=2000, scale=500, size=num_repos).clip(100, 10000).astype(np.float32)
    doc_quality = np.random.uniform(0.1, 1.0, size=num_repos).astype(np.float32)
    code_health = np.random.uniform(0.3, 1.0, size=num_repos).astype(np.float32)
    activity_score = np.random.uniform(0.0, 1.0, size=num_repos).astype(np.float32)
    trend_velocity = np.random.exponential(scale=0.1, size=num_repos).clip(0, 1).astype(np.float32)
    
    # 3. Simulate Interactions
    num_interactions = 500000
    user_indices = np.random.randint(0, num_users, size=num_interactions)
    repo_indices = np.random.randint(0, num_repos, size=num_interactions)
    
    print("Computing base affinities and labels...")
    batch_users = user_embs[user_indices] # 768 MB in float32
    batch_repos = repo_embs[repo_indices] # 768 MB in float32
    
    # Highly memory-efficient dot product
    base_affinity = np.einsum('ij,ij->i', batch_users, batch_repos)
    
    # Extract the batch arrays for the metadata
    batch_stars = star_counts[repo_indices]
    batch_forks = fork_counts[repo_indices]
    batch_issues = open_issues[repo_indices]
    batch_pushed = pushed_days_ago[repo_indices]
    batch_readme = readme_lengths[repo_indices]
    batch_doc = doc_quality[repo_indices]
    batch_health = code_health[repo_indices]
    batch_activity = activity_score[repo_indices]
    batch_trend = trend_velocity[repo_indices]
    
    # Skill match score (Cross-feature calculated on the fly in production, we fake it here)
    # We correlate it slightly with semantic affinity for realism
    skill_match_score = np.clip(base_affinity * 0.5 + np.random.normal(0.5, 0.2, size=num_interactions), 0, 1).astype(np.float32)
    
    # FEATURE_ORDER column mapping: doc_quality=batch_doc,
    # code_health=batch_health, readme_length=batch_readme,
    # star_count=batch_stars, fork_count=batch_forks,
    # open_issues_count=batch_issues, pushed_days_ago=batch_pushed,
    # activity_score=batch_activity, trend_velocity=batch_trend,
    # skill_match_score=skill_match_score.
    dense_columns = {
        "doc_quality": batch_doc,
        "code_health": batch_health,
        "readme_length": batch_readme,
        "star_count": batch_stars,
        "fork_count": batch_forks,
        "open_issues_count": batch_issues,
        "pushed_days_ago": batch_pushed,
        "activity_score": batch_activity,
        "trend_velocity": batch_trend,
        "skill_match_score": skill_match_score,
    }
    batch_dense = np.column_stack(
        [dense_columns[name] for name in FEATURE_ORDER]
    ).astype(np.float32)
    
    # 4. Generate the 5 Labels (Now influenced by the new features!)
    # Clicks influenced by trend velocity and skill match
    ctr_prob = 1 / (1 + np.exp(-(base_affinity * 3 + (skill_match_score * 4) + (batch_trend * 3) - 1)))
    ctr_label = (np.random.rand(num_interactions) < ctr_prob).astype(np.float32)
    
    # Saves influenced by code health and skill match
    save_prob = 1 / (1 + np.exp(-(skill_match_score * 5 + batch_health * 3 - 2)))
    save_label = ((np.random.rand(num_interactions) < save_prob) & (ctr_label == 1)).astype(np.float32)
    
    # GH Open influenced by recent activity and open issues
    gh_prob = 1 / (1 + np.exp(-(base_affinity * 2 + batch_activity * 4 - (batch_pushed / 50))))
    gh_label = ((np.random.rand(num_interactions) < gh_prob) & (ctr_label == 1)).astype(np.float32)
    
    # Dwell time influenced heavily by doc quality and readme length
    dwell_time = 10 + (base_affinity * 50) + (batch_doc * 100) + (batch_readme / 50) + (save_label * 120) + np.random.normal(0, 10, num_interactions)
    dwell_label = (np.clip(dwell_time, 0, 600) / 600.0).astype(np.float32)
    
    # Follows influenced by extreme quality and skill match
    follow_prob = 1 / (1 + np.exp(-(skill_match_score * 6 + batch_health * 4 - 4)))
    follow_label = ((np.random.rand(num_interactions) < follow_prob) & (save_label == 1)).astype(np.float32)
    
    print("Saving to NPZ format...")
    np.savez_compressed(
        output_file,
        user_embs=batch_users,
        repo_embs=batch_repos,
        dense_features=batch_dense,
        y_ctr=ctr_label,
        y_save=save_label,
        y_gh=gh_label,
        y_dwell=dwell_label,
        y_follow=follow_label
    )
    print(f"✅ Successfully saved dataset as {output_file}! Total size: {num_interactions} rows.")

if __name__ == "__main__":
    out_path = "training_data.npz"
    generate_synthetic_data(num_users=1000, num_repos=10000, emb_dim=384, output_file=out_path)
