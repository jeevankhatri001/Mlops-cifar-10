import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

DB_USER     = os.getenv("MARIADB_USER", "mlops")
DB_PASSWORD = os.getenv("MARIADB_PASSWORD", "")
DB_NAME     = os.getenv("MARIADB_DB", "mlops_db")
DB_HOST     = os.getenv("MARIADB_HOST", "localhost")
DB_PORT     = os.getenv("MARIADB_PORT", "3306")

_user = quote_plus(DB_USER)
_pw   = quote_plus(DB_PASSWORD)
_auth = _user if not DB_PASSWORD else f"{_user}:{_pw}"

# PyMySQL driver — pure Python, no compilation needed
DATABASE_URL = f"mysql+pymysql://{_auth}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(schema_path=None):
    if schema_path is None:
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r") as f:
        ddl = f.read()
    # Run each statement individually
    statements = [s.strip() for s in ddl.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.exec_driver_sql(stmt)
    print(f"Schema applied from {schema_path}")


if __name__ == "__main__":
    init_db()
    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT table_name FROM information_schema.tables "
                 "WHERE table_schema = :db ORDER BY table_name"),
            {"db": DB_NAME},
        ).fetchall()
    print("Tables in database:", [t[0] for t in tables])
