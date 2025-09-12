import pandas as pd
import psycopg2
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
file_path = DATA_DIR / "hospital-data-analysis.csv"

df = pd.read_csv(file_path)

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host="127.0.0.1",
    port="5432"
)
cur = conn.cursor()

for _, row in df.iterrows():
    cur.execute("""
        INSERT INTO patient_records ("Age", "Gender", "Condition", "Procedure", "Cost", "Length_of_Stay", "Readmission", "Outcome", "Satisfaction")
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        row['Age'], row['Gender'], row['Condition'], row['Procedure'],
        row['Cost'], row['Length_of_Stay'], row['Readmission'],
        row['Outcome'], row['Satisfaction']
    ))

conn.commit()
cur.close()
conn.close()