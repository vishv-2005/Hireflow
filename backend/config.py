# config.py - backend configuration for deployment
import os

# We use environment variables instead of hardcoding URLs so that the
# code can run on any environment (localhost or the EC2 server) without
# needing to be changed.
BACKEND_URL = os.environ.get("HIREFLOW_BACKEND_URL", "http://localhost:5001")

# Future: PostgreSQL on AWS RDS
# DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://user:pass@host:5432/hireflow")
