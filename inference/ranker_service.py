import torch
import torch.nn as nn
import numpy as np
import json
import os

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
    def __init__(self, model_path="heavy_ranker.pt", scaler_path="feature_scaler.json", emb_dim=384):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emb_dim = emb_dim
        
        # Total input dim = User_emb(384) + Repo_emb(384) + DenseFeatures(10)
        self.input_dim = (emb_dim * 2) + 10
        
        # Load Model
        self.model = MMoEHeavyRanker(self.input_dim).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print("✅ Heavy Ranker Model loaded successfully.")
        else:
            print(f"⚠️ WARNING: {model_path} not found. Running with untrained weights for demo.")
        self.model.eval()
            
        # Load Scaler safely from JSON
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler_params = json.load(f)
            self.scaler_mean = np.array(scaler_params['mean'], dtype=np.float32)
            self.scaler_scale = np.array(scaler_params['scale'], dtype=np.float32)
            print("✅ Feature Scaler JSON loaded successfully.")
        else:
            raise RuntimeError(f"❌ ERROR: {scaler_path} not found! Refusing to start with unscaled features which corrupt predictions.")

    def score_batch(self, user_embedding, user_skills, candidate_repos):
        """
        Executes the MMoE network on a micro-batch (e.g. 15 or 150 repos).
        """
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
            row = [
                repo.get('doc_quality', 0.5),
                repo.get('code_health', 0.5),
                repo.get('readme_length', 1000),
                repo.get('star_count', 0),
                repo.get('fork_count', 0),
                repo.get('open_issues_count', 0),
                repo.get('pushed_days_ago', 365),
                repo.get('activity_score', 0.0),
                repo.get('trend_velocity', 0.0),
                skill_match
            ]
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

        # 3. Apply The Value Function (Business Math)
        # Weights: Clicks=1, Save=5, GH_Open=2, Dwell=0.1, Follow=20
        final_scores = (1.0 * p_ctr) + (5.0 * p_save) + (2.0 * p_gh) + (0.1 * p_dwell) + (20.0 * p_fol)
        
        # 4. Attach scores and sort
        results = []
        for i, repo in enumerate(candidate_repos):
            results.append({
                "repo_id": repo['id'],
                "final_score": float(final_scores[i]),
                "skill_match": float(unscaled_features[i][9]), # For debug
                "predictions": {
                    "p_ctr": float(p_ctr[i]),
                    "p_save": float(p_save[i]),
                    "p_gh": float(p_gh[i]),
                    "pred_dwell_fraction": float(p_dwell[i]), # Multiply by 600 to get seconds
                    "p_follow": float(p_fol[i])
                }
            })
            
        # Sort descending by score
        results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        return results



# ------------------------------ Demonstration of Micro-Batching Logic ----------------------------------------
# !!!!!!!!!!!!----------------------------will be removed later-------------------------------!!!!!!!!!!!!!!

if __name__ == "__main__":
    print("\n--- Testing Ranker Service with Rich Metadata ---")
    service = RankerService()
    
    # Fake user data
    current_user_embedding = np.random.randn(384)
    current_user_embedding /= np.linalg.norm(current_user_embedding)
    user_skills = ["Python", "Machine Learning", "PyTorch", "AI/ML"]
    
    # Fake pool of 150 candidates from Qdrant
    pool = []
    for i in range(150):
        # We simulate one really good Python repo
        is_perfect = (i == 42)
        pool.append({
            'id': f'repo_{i}',
            'embedding': np.random.randn(384) / np.linalg.norm(np.random.randn(384)),
            'doc_quality': 0.9 if is_perfect else 0.4,
            'code_health': 0.8 if is_perfect else 0.5,
            'readme_length': 5000 if is_perfect else 800,
            'star_count': 10000 if is_perfect else np.random.randint(10, 500),
            'fork_count': 2000 if is_perfect else np.random.randint(0, 100),
            'open_issues_count': 50 if is_perfect else np.random.randint(0, 20),
            'pushed_days_ago': 1 if is_perfect else np.random.randint(30, 365),
            'activity_score': 0.9 if is_perfect else 0.1,
            'trend_velocity': 0.8 if is_perfect else 0.05,
            'languages': ["Python"] if is_perfect else ["Java"],
            'topics': ["Machine Learning", "AI"] if is_perfect else ["enterprise"],
            'tags': ["PyTorch"] if is_perfect else ["spring-boot"]
        })
        
    print(f"Candidate Pool Size: {len(pool)} repos from Qdrant")
    
    # ---------------------------------------------------------
    # STEP 1: INITIAL TRUE RANKING (Score all 150)
    # ---------------------------------------------------------
    print("\nExecuting Heavy Ranker on ALL 150 candidates...")
    ranked_pool = service.score_batch(current_user_embedding, user_skills, pool)
    
    # Serve Batch 1 (Top 15)
    feed_batch_1 = ranked_pool[:15]
    print("\n--- SERVING BATCH 1 (Top 15) ---")
    for r in feed_batch_1[:3]: # Just printing top 3 for brevity
        print(f"  {r['repo_id']} | Final Score: {r['final_score']:.2f} | Skill Match: {r['skill_match']:.2f}")
        
    # ---------------------------------------------------------
    # STEP 2: REAL-TIME FEEDBACK
    # ---------------------------------------------------------
    print("\n[User interacts with Batch 1 -> Real-time Feedback Loop triggers]")
    # User's embedding is updated slightly based on interaction
    current_user_embedding += np.random.randn(384) * 0.1
    current_user_embedding /= np.linalg.norm(current_user_embedding)
    
    # We remove the 15 repos we already showed them
    seen_repo_ids = {r['repo_id'] for r in feed_batch_1}
    remaining_pool = [r for r in pool if r['id'] not in seen_repo_ids]
    
    # ---------------------------------------------------------
    # STEP 3: RE-RANKING (Score the remaining 135)
    # ---------------------------------------------------------
    print(f"\nRe-Executing Heavy Ranker on remaining {len(remaining_pool)} candidates...")
    ranked_pool_2 = service.score_batch(current_user_embedding, user_skills, remaining_pool)
    
    # Serve Batch 2 (Next Top 15)
    feed_batch_2 = ranked_pool_2[:15]
    print("\n--- SERVING BATCH 2 (Top 15) ---")
    for r in feed_batch_2[:3]:
        print(f"  {r['repo_id']} | Final Score: {r['final_score']:.2f}")

