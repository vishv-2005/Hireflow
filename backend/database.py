# database.py - handles all database connections and queries
import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Get the database URL from the environment, just like we did in config.py.
# This defaults to a local SQLite database file for easy development.
# In production, AWS RDS will inject a MySQL URL here.
# Vercel serverless restricts write access to /tmp.
if os.environ.get("VERCEL"):
    default_db_url = "sqlite:////tmp/hireflow_dev.db"
else:
    default_db_url = "sqlite:///hireflow_dev.db"

DATABASE_URL = os.environ.get("DATABASE_URL", default_db_url)

# 2. Set up SQLAlchemy engine and session
# The "Engine" is what actually talks to the database.
# The "Session" is our temporary workspace where we build our queries before committing them.
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# "Base" is a special class that our ORM models will inherit from
Base = declarative_base()

# 3. Define the Candidate model
# Object-Relational Mapping (ORM) lets us write Python classes instead of raw SQL queries.
class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # We use a batch_id (a UUID string) to group together all candidates
    # that were uploaded in the same ZIP file. This way we know which results
    # belong to which upload!
    batch_id = Column(String(36), index=True, nullable=False)
    
    name = Column(String(255), nullable=False)
    score = Column(Float, nullable=False)
    
    # Databases don't have a native "List" type, so we convert the Python
    # list of matched skills into a JSON string to store it in a Text column.
    matched_skills = Column(Text, nullable=False)
    
    filename = Column(String(255), nullable=False)
    is_anomaly = Column(Boolean, default=False)
    anomaly_reason = Column(Text, nullable=True)
    rank = Column(Integer, nullable=False)
    
    # Automatically recording when this row was created is a great practice.
    created_at = Column(DateTime, default=datetime.utcnow)


# 4. Initialize Database
def init_db():
    """
    Creates the 'candidates' table in the database if it doesn't exist yet.
    """
    Base.metadata.create_all(bind=engine)


# 5. Save candidates to db
def save_candidates(candidates_list, batch_id):
    """
    Takes the ranked list of dictionary candidates (from our scorer) 
    and saves them as Candidate ORM objects in the database.
    """
    session = SessionLocal()
    try:
        # Loop through our dicts and create Database Objects
        for cand_dict in candidates_list:
            db_candidate = Candidate(
                batch_id=batch_id,
                name=cand_dict["name"],
                score=cand_dict["score"],
                # Convert list to JSON string before saving
                matched_skills=json.dumps(cand_dict["matched_skills"]),
                filename=cand_dict["filename"],
                is_anomaly=cand_dict.get("is_anomaly", False),
                anomaly_reason=cand_dict.get("anomaly_reason", ""),
                rank=cand_dict["rank"]
            )
            session.add(db_candidate)
        
        # Commit all changes at once
        session.commit()
    except Exception as e:
        # If anything fails, rollback so we don't save partial/corrupted batches
        session.rollback()
        print(f"Failed to save candidates to DB: {e}")
        raise e
    finally:
        # Always close the session so we don't leak database connections!
        session.close()


# 6. Retrieve latest batch
def get_latest_batch():
    """
    Finds the most recently uploaded batch and returns those candidates
    so the frontend dashboard can display them.
    We return plain dictionaries, not ORM objects, so Flask can JSONify them.
    """
    session = SessionLocal()
    try:
        # 1. Find the most recently created candidate row to figure out its batch_id
        latest_record = session.query(Candidate).order_by(Candidate.created_at.desc()).first()
        
        if not latest_record:
            return [] # DB is completely empty
            
        latest_batch_id = latest_record.batch_id
        
        # 2. Get all candidates that share that newest batch_id, sorted by rank
        batch_candidates = session.query(Candidate)\
            .filter(Candidate.batch_id == latest_batch_id)\
            .order_by(Candidate.rank.asc())\
            .all()
            
        # 3. Convert ORM objects back into plain dicts for the frontend API
        result = []
        for cand in batch_candidates:
            result.append({
                "name": cand.name,
                "score": cand.score,
                # Convert the JSON string back into a Python List
                "matched_skills": json.loads(cand.matched_skills),
                "filename": cand.filename,
                "is_anomaly": cand.is_anomaly,
                "anomaly_reason": cand.anomaly_reason or "",
                "rank": cand.rank
            })
            
        return result
    finally:
        session.close()
