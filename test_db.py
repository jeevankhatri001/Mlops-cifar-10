from sqlalchemy import text

from app.database.db import engine

try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))

        for row in result:
            print(row)

    print("Database connection successful!")

except Exception as e:
    print("Connection failed:")
    print(e)
