# Heavy Ranker Architecture Specification

## 1. Overview
The Heavy Ranker is a PyTorch-based Multi-gate Mixture-of-Experts (MMoE) Neural Network. It acts as the second stage in a Two-Stage Recommendation Pipeline, taking a candidate pool of 150 repos from the Vector Database (Light Ranker) and scoring them in real-time.

## 2. Input Tensor (Dimension: 778)
The network requires a single concatenated tensor of size 778, consisting of 3 core blocks:

### A. Semantic Embeddings (768 dims)
- `user_embedding`: 384-dimensional dense float array (Tracks semantic intent)
- `repo_embedding`: 384-dimensional dense float array (Tracks semantic content)

### B. Dense Metadata Features (10 dims)
*Note: All 10 features must be processed using a `StandardScaler` (to normalize them to a standard distribution) before entering the network.*

**Quality & Health Metrics**
1. `doc_quality` (0.0 to 1.0 scale)
2. `code_health` (0.0 to 1.0 scale)
3. `readme_length` (Integer word/character count)

**Popularity & Engagement Metrics**
4. `star_count` (Integer)
5. `fork_count` (Integer)
6. `open_issues_count` (Integer)

**Freshness & Momentum Metrics**
7. `pushed_days_ago` (Integer days)
8. `activity_score` (Float)
9. `trend_velocity` (Float)

**Dynamic Cross-Features**
10. `skill_match_score` (Float percentage 0.0-1.0)
    - *Tweak Note:* This is entirely stateless and calculated on the fly in the backend API (FastAPI/Node). It performs a fast Set Intersection between the User's `interests/skills` string arrays and the Repo's native `languages/topics/tags` Qdrant string arrays.

## 3. Network Outputs (5 Tasks)
The MMoE architecture routes the 778-dim input through 4 Shared Experts, and outputs 5 distinct predictions via 5 Task Heads:

1. `Click Probability (CTR)` (Binary Classification -> Sigmoid, 0.0 to 1.0)
2. `Save Probability` (Binary Classification -> Sigmoid, 0.0 to 1.0)
3. `GitHub Open Probability` (Binary Classification -> Sigmoid, 0.0 to 1.0)
4. `Follow Probability` (Binary Classification -> Sigmoid, 0.0 to 1.0)
5. `Dwell Time` (Regression -> ReLU, normalized fraction 0.0 to 1.0, where 1.0 = 600s)

## 4. The Value Function (Final Ranking)
The 5 outputs are fed into a hard-coded algebraic equation in the backend to calculate the final rank `Score`. The 150 candidate repos are sorted descending by this score before being chopped into batches of 15 for the frontend.

`Score = (w1 * p_ctr) + (w2 * p_save) + (w3 * p_gh) + (w4 * p_fol) + (w5 * p_dwell)`

## 5. The Batching Architecture (True Ranking vs Micro-Sorting)
To ensure the user gets the absolute best content, the Heavy Ranker must evaluate the entire candidate pool retrieved by the Vector Database before serving the user.

**The Production Flow:**
1. **Retrieval:** Qdrant retrieves 150 candidate repos based purely on semantic similarity.
2. **Initial True Ranking:** The backend passes *all 150 repos* through the Neural Network simultaneously. This takes <5ms and sorts them by actual quality (Stars, Health, Skill Match), rescuing "hidden gems" that Qdrant may have placed at the bottom.
3. **Serve Batch 1:** The backend slices the Top 15 repos from the newly sorted list and sends them to the frontend.
4. **Real-Time Update:** As the user interacts with Batch 1, their semantic User Embedding is updated.
5. **Re-Ranking:** The backend takes the remaining 135 unseen repos and runs them through the Neural Network again using the *new* User Embedding. 
6. **Serve Batch 2:** The newly re-sorted Top 15 are served to the user, reflecting their most recent interactions in real-time.
