import os
import logging
import json
import re
from urllib.parse import urlparse
from dotenv import load_dotenv

# Import connector
from database.connector import PostgreSQLConnector, _SUPABASE_HOST_RE

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
logger = logging.getLogger("db.sync")

def sync():
    load_dotenv()
    
    # 1. Connect to Local PostgreSQL
    local_url = os.getenv("LOCAL_DATABASE_URL")
    if not local_url:
        db_url = os.getenv("DATABASE_URL")
        # Check if database_url is local (i.e. not Supabase)
        is_supabase = False
        if db_url:
            try:
                hostname = urlparse(db_url).hostname or ""
                is_supabase = bool(_SUPABASE_HOST_RE.search(hostname))
            except Exception:
                pass
        
        if db_url and not is_supabase:
            local_url = db_url
        else:
            local_url = "postgresql://medhanshadhlakha@localhost:5432/gh_social"
            
    # Mask password for secure logging
    try:
        parsed = urlparse(local_url)
        masked_local = f"{parsed.scheme}://{parsed.username or ''}@{parsed.hostname or 'localhost'}:{parsed.port or 5432}{parsed.path}"
    except Exception:
        masked_local = "local PostgreSQL"

    logger.info(f"Connecting to Local PostgreSQL: {masked_local}")
    local_db = PostgreSQLConnector(database_url=local_url)
    if not local_db.verify_connection():
        logger.error("Failed to connect to local database.")
        return
    # Ensure local database schema is fully initialized/upgraded before querying
    try:
        local_db.init_db()
    except Exception as exc:
        logger.warning(f"Local database schema initialization warning: {exc}")
        
    # 2. Connect to Supabase
    supabase_url = os.getenv("SUPABASE_DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
    if not supabase_url:
        # Fall back to DATABASE_URL if it points to Supabase
        db_url = os.getenv("DATABASE_URL")
        is_supabase = False
        if db_url:
            try:
                hostname = urlparse(db_url).hostname or ""
                is_supabase = bool(_SUPABASE_HOST_RE.search(hostname))
            except Exception:
                pass
        if is_supabase:
            supabase_url = db_url
            
    if not supabase_url:
        logger.error("Neither SUPABASE_DATABASE_URL nor a Supabase DATABASE_URL environment variable is set.")
        return
        
    try:
        parsed = urlparse(supabase_url)
        masked_sb = f"{parsed.scheme}://{parsed.username or ''}@{parsed.hostname or ''}:{parsed.port or 6543}{parsed.path}"
    except Exception:
        masked_sb = "Supabase PostgreSQL"
        
    logger.info(f"Connecting to Supabase PostgreSQL: {masked_sb}")
    supabase_db = PostgreSQLConnector(database_url=supabase_url)
    if not supabase_db.verify_connection():
        logger.error("Failed to connect to Supabase database.")
        return
        
    # 3. Read rows from Local PostgreSQL
    logger.info("Reading repositories from local database...")
    local_conn = local_db.connect()
    local_cursor = local_conn.cursor()
    
    columns = [
        "repo_id", "github_repo_url", "owner_id", "repo_name", "full_name",
        "description", "primary_language", "language_used", "topics",
        "readme_summary", "star_count", "likes_count", "comments_count",
        "saves_count", "views_count", "forks_count", "pr_count",
        "created_at", "updated_at"
    ]
    
    query = f"SELECT {', '.join(columns)} FROM Repo;"
    local_cursor.execute(query)
    rows = local_cursor.fetchall()
    local_conn.close()
    
    logger.info(f"Retrieved {len(rows)} repositories from local database.")
    if not rows:
        logger.warning("No records to sync.")
        return
        
    # 4. Write/Upsert to Supabase
    logger.info("Initializing Supabase database schemas if not exist...")
    supabase_db.init_db()
    
    supabase_conn = supabase_db.connect()
    supabase_cursor = supabase_conn.cursor()
    
    logger.info("Syncing records to Supabase...")
    
    count = 0
    batch_size = 50
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            params = []
            values_clauses = []
            for row in batch:
                row_list = list(row)
                # Parse JSON columns if they are not already parsed dicts/lists
                # (column idx 7 is language_used, 8 is topics)
                for idx in [7, 8]:
                    if isinstance(row_list[idx], (dict, list)):
                        row_list[idx] = json.dumps(row_list[idx])
                params.extend(row_list)
                
                placeholders = [
                    "CAST(%s AS jsonb)" if col in ("language_used", "topics") else "%s"
                    for col in columns
                ]
                values_clauses.append(f"({', '.join(placeholders)})")
                
            upsert_query = f"""
            INSERT INTO Repo ({', '.join(columns)})
            VALUES {', '.join(values_clauses)}
            ON CONFLICT (github_repo_url) DO UPDATE SET
                repo_name = EXCLUDED.repo_name,
                full_name = EXCLUDED.full_name,
                description = EXCLUDED.description,
                primary_language = EXCLUDED.primary_language,
                language_used = EXCLUDED.language_used,
                topics = EXCLUDED.topics,
                readme_summary = EXCLUDED.readme_summary,
                star_count = EXCLUDED.star_count,
                likes_count = EXCLUDED.likes_count,
                comments_count = EXCLUDED.comments_count,
                saves_count = EXCLUDED.saves_count,
                views_count = EXCLUDED.views_count,
                forks_count = EXCLUDED.forks_count,
                pr_count = EXCLUDED.pr_count,
                updated_at = EXCLUDED.updated_at;
            """
            supabase_cursor.execute(upsert_query, params)
            supabase_conn.commit()
            count += len(batch)
            logger.info(f"  Synced {count}/{len(rows)} rows...")
        except Exception as exc:
            supabase_conn.rollback()
            logger.error(f"Failed to sync batch starting at index {i}: {exc}")
            
    supabase_conn.close()
    
    # 5. Verify Supabase Count
    logger.info("Verification: checking total count in Supabase...")
    final_count = supabase_db.get_repo_count()
    logger.info(f"Verification complete! Supabase database has {final_count} repositories.")

if __name__ == "__main__":
    sync()
