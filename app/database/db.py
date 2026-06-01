import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
DB_NAME = os.getenv("POSTGRES_DB", "mlops_db")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# Build the connection URL (also works for password-less 'trust' auth)
_user = quote_plus(DB_USER)
_pw = quote_plus(DB_PASSWORD)
_auth = _user if not DB_PASSWORD else f"{_user}:{_pw}"
DATABASE_URL = f"postgresql+psycopg2://{_auth}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(schema_path=None):
    """Create all tables defined in schema.sql (idempotent)."""
    if schema_path is None:
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r") as f:
        ddl = f.read()
    with engine.begin() as conn:
        conn.exec_driver_sql(ddl)
    print(f"Schema applied from {schema_path}")


if __name__ == "__main__":
    init_db()
    with engine.connect() as conn:
        tables = conn.execute(text(
            "SELECT tablename FROM pg_tables "
            "WHERE schemaname='public' ORDER BY tablename"
        )).fetchall()
    print("Tables in database:", [t[0] for t in tables])
