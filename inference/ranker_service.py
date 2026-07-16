import torch
import torch.nn as nn
import numpy as np
import json
import logging
import os

from inference.feature_spec import (
    EMBEDDING_DIM,
    FEATURE_COUNT,
    FEATURE_ORDER,
    FEATURE_SPEC_VERSION,
    INPUT_DIM,
    RANKER_MODEL_VERSION,
)
from inference.value_function import VALUE_WEIGHTS, compute_value_score
from config import REPOSITORY_EMBEDDING_VERSION


logger = logging.getLogger("pipeline.ranker")

class MMoEHeavyRanker(nn.Module):
    def __init__(self, input_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = 5
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU()
            ) for _ in range(self.num_experts)
        ])
        
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, self.num_experts), nn.Softmax(dim=1))
            for _ in range(self.num_tasks)
        ])
        
        self.head_ctr = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_save = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_gh = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_dwell = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.ReLU())
        self.head_follow = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        gate_ctr = self.gates[0](x).unsqueeze(2)
        out_ctr = self.head_ctr(torch.sum(expert_outputs * gate_ctr, dim=1)).squeeze()
        
        gate_save = self.gates[1](x).unsqueeze(2)
        out_save = self.head_save(torch.sum(expert_outputs * gate_save, dim=1)).squeeze()
        
        gate_gh = self.gates[2](x).unsqueeze(2)
        out_gh = self.head_gh(torch.sum(expert_outputs * gate_gh, dim=1)).squeeze()
        
        gate_dwell = self.gates[3](x).unsqueeze(2)
        out_dwell = self.head_dwell(torch.sum(expert_outputs * gate_dwell, dim=1)).squeeze()
        
        gate_follow = self.gates[4](x).unsqueeze(2)
        out_follow = self.head_follow(torch.sum(expert_outputs * gate_follow, dim=1)).squeeze()
        
        return out_ctr, out_save, out_gh, out_dwell, out_follow

def calculate_match_score(user_interests_skills, repo_languages, repo_topics, repo_tags):
    """Calculates the percentage of overlapping keywords dynamically."""
    if not user_interests_skills:
        return 0.0
    
    # Combine all repo text arrays into one set of lowercase keywords
    repo_keywords = set([str(w).lower() for w in repo_languages + repo_topics + repo_tags])
    
    matches = 0
    for skill in user_interests_skills:
        if str(skill).lower() in repo_keywords:
            matches += 1
            
    return matches / len(user_interests_skills)

class RankerService:
    def __init__(
        self,
        model_path="heavy_ranker.pt",
        scaler_path="feature_scaler.json",
        emb_dim=384,
        manifest_path=None,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emb_dim = EMBEDDING_DIM
        self.model_version = RANKER_MODEL_VERSION
        self.embedding_version = REPOSITORY_EMBEDDING_VERSION
        self.feature_spec_version = FEATURE_SPEC_VERSION
        
        # Total input dim = User_emb(384) + Repo_emb(384) + DenseFeatures(10)
        self.input_dim = INPUT_DIM

        if manifest_path is None:
            manifest_path = os.path.join(
                os.path.dirname(os.path.abspath(model_path)),
                "model_manifest.json",
            )
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            expected_contract = {
                "input_dim": INPUT_DIM,
                "embedding_dim": EMBEDDING_DIM,
                "feature_count": FEATURE_COUNT,
                "feature_spec_version": FEATURE_SPEC_VERSION,
            }
            mismatches = {
                key: (manifest.get(key), expected)
                for key, expected in expected_contract.items()
                if manifest.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    f"Model manifest is incompatible with inference: {mismatches}"
                )
            self.model_version = str(
                manifest.get("model_version") or RANKER_MODEL_VERSION
            )
            self.embedding_version = str(
                manifest.get("embedding_version")
                or REPOSITORY_EMBEDDING_VERSION
            )
            logger.info("Model manifest loaded successfully from %s", manifest_path)
        
        # Load Model
        self.model = MMoEHeavyRanker(self.input_dim).to(self.device)
        self._model_loaded = False
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self._model_loaded = True
            logger.info("Heavy Ranker model loaded successfully from %s", model_path)
        else:
            logger.warning("Heavy Ranker model not found at %s", model_path)
        self.model.eval()
            
        # Load Scaler safely from JSON
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler_params = json.load(f)
            self.scaler_mean = np.array(scaler_params['mean'], dtype=np.float32)
            self.scaler_scale = np.array(scaler_params['scale'], dtype=np.float32)
            if len(self.scaler_mean) != FEATURE_COUNT:
                raise ValueError(
                    f"Feature scaler has {len(self.scaler_mean)} features; "
                    f"expected {FEATURE_COUNT}"
                )
            logger.info("Feature scaler loaded successfully from %s", scaler_path)
        else:
            raise RuntimeError(f"{scaler_path} not found; refusing to use unscaled features")

    def score_batch(self, user_embedding, user_skills, candidate_repos):
        """
        Executes the MMoE network on a micro-batch (e.g. 15 or 150 repos).
        """
        if not self._model_loaded:
            logger.warning("Heavy Ranker model is not loaded; returning no scores")
            return []

        if not candidate_repos:
            return []
            
        # 1. Prepare inputs
        user_embs = np.tile(user_embedding, (len(candidate_repos), 1))
        repo_embs = np.vstack([repo['embedding'] for repo in candidate_repos])
        
        dense_features = []
        for repo in candidate_repos:
            # Dynamically calculate the cross-feature!
            skill_match = calculate_match_score(user_skills, repo.get('languages', []), repo.get('topics', []), repo.get('tags', []))
            
            # The EXACT 10 features generated in DataGen:
            # batch_doc, batch_health, batch_readme, batch_stars, batch_forks, batch_issues, batch_pushed, batch_activity, batch_trend, skill_match_score
            feature_values = {
                'doc_quality': repo.get('doc_quality', 0.5),
                'code_health': repo.get('code_health', 0.5),
                'readme_length': repo.get('readme_length', 1000),
                'star_count': repo.get('star_count', 0),
                'fork_count': repo.get('fork_count', 0),
                'open_issues_count': repo.get('open_issues_count', 0),
                'pushed_days_ago': repo.get('pushed_days_ago', 365),
                'activity_score': repo.get('activity_score', 0.0),
                'trend_velocity': repo.get('trend_velocity', 0.0),
                'skill_match_score': skill_match,
            }
            row = [feature_values[name] for name in FEATURE_ORDER]
            dense_features.append(row)
            
        dense_features = np.array(dense_features)
        unscaled_features = dense_features.copy()
        
        # Manually apply StandardScaler math
        dense_features = (dense_features - self.scaler_mean) / self.scaler_scale
            
        # Concatenate into massive input tensor
        X = np.hstack((user_embs, repo_embs, dense_features))
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # 2. Run Heavy Ranker Inference
        with torch.no_grad():
            p_ctr, p_save, p_gh, p_dwell, p_fol = self.model(X_tensor)
            
            # Ensure 1D array handling
            if len(candidate_repos) == 1:
                p_ctr, p_save, p_gh, p_dwell, p_fol = p_ctr.unsqueeze(0), p_save.unsqueeze(0), p_gh.unsqueeze(0), p_dwell.unsqueeze(0), p_fol.unsqueeze(0)
                
            p_ctr = p_ctr.cpu().numpy()
            p_save = p_save.cpu().numpy()
            p_gh = p_gh.cpu().numpy()
            p_dwell = p_dwell.cpu().numpy()
            p_fol = p_fol.cpu().numpy()

        # 3. Apply the value function and attach scores.
        results = []
        for i, repo in enumerate(candidate_repos):
            predictions_dict = {
                "p_ctr": float(p_ctr[i]),
                "p_save": float(p_save[i]),
                "p_gh": float(p_gh[i]),
                "pred_dwell_fraction": float(p_dwell[i]),
                "p_follow": float(p_fol[i]),
            }
            results.append({
                "repo_id": repo['id'],
                "final_score": compute_value_score(predictions_dict),
                "skill_match": float(unscaled_features[i][9]), # For debug
                "predictions": predictions_dict,
            })
            
        # Sort descending by score
        results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        return results
