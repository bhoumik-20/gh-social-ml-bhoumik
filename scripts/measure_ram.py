#!/usr/bin/env python3
"""Measure RAM usage during app startup — simulates what happens on Render/Railway."""
import os, sys, tracemalloc, psutil

# Apply the same thread limits as api/main.py
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

from dotenv import load_dotenv
load_dotenv()

process = psutil.Process(os.getpid())

def mb(bytes_val):
    return f"{bytes_val / (1024 * 1024):.1f} MB"

print("=" * 60)
print("RAM USAGE PROFILER — Deployment Simulation")
print("=" * 60)

# Baseline
baseline = process.memory_info().rss
print(f"\n🟢 Baseline (Python + dotenv):          {mb(baseline)}")

# Step 1: FastAPI + Pydantic
from fastapi import FastAPI
from pydantic import BaseModel
after_fastapi = process.memory_info().rss
print(f"📦 After FastAPI + Pydantic:             {mb(after_fastapi)}  (+{mb(after_fastapi - baseline)})")

# Step 2: PyTorch (biggest offender)
try:
    import torch
    after_torch = process.memory_info().rss
    print(f"🔥 After PyTorch import:                 {mb(after_torch)}  (+{mb(after_torch - after_fastapi)})")
except ImportError:
    after_torch = process.memory_info().rss
    print(f"⚠️  PyTorch not installed, skipped.")

# Step 3: SentenceTransformers (loads the embedding model)
try:
    from sentence_transformers import SentenceTransformer
    after_st_import = process.memory_info().rss
    print(f"📦 After SentenceTransformers import:    {mb(after_st_import)}  (+{mb(after_st_import - after_torch)})")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    after_model = process.memory_info().rss
    print(f"🧠 After loading all-MiniLM-L6-v2:       {mb(after_model)}  (+{mb(after_model - after_st_import)})")
except ImportError:
    after_model = process.memory_info().rss
    print(f"⚠️  SentenceTransformers not installed, skipped.")

# Step 4: Database connector
try:
    from database import PostgreSQLConnector
    after_db = process.memory_info().rss
    print(f"📦 After Database connector:             {mb(after_db)}  (+{mb(after_db - after_model)})")
except Exception:
    after_db = after_model

# Step 5: Qdrant client
try:
    from qdrant_client import QdrantClient
    after_qdrant = process.memory_info().rss
    print(f"📦 After Qdrant client:                  {mb(after_qdrant)}  (+{mb(after_qdrant - after_db)})")
except ImportError:
    after_qdrant = after_db
    print(f"⚠️  Qdrant client not installed, skipped.")

# Step 6: Redis (feedback queue)
try:
    import redis
    after_redis = process.memory_info().rss
    print(f"📦 After Redis client:                   {mb(after_redis)}  (+{mb(after_redis - after_qdrant)})")
except ImportError:
    after_redis = after_qdrant
    print(f"⚠️  Redis not installed, skipped.")

# Step 7: Requests (for OpenRouter API calls)
import requests
after_requests = process.memory_info().rss
print(f"📦 After Requests (OpenRouter HTTP):     {mb(after_requests)}  (+{mb(after_requests - after_redis)})")

# Final
final = process.memory_info().rss
print(f"\n{'=' * 60}")
print(f"🏁 TOTAL RAM at full startup:            {mb(final)}")
print(f"{'=' * 60}")

# Render free tier is 512 MB, paid starter is 2 GB
if final < 512 * 1024 * 1024:
    print("✅ Fits within Render FREE tier (512 MB)")
elif final < 2048 * 1024 * 1024:
    print("⚠️  Needs Render STARTER tier (2 GB) or equivalent")
else:
    print("❌ Exceeds 2 GB — needs a larger instance")
