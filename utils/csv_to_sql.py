import pandas as pd
import psycopg2
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
file_path = DATA_DIR / "hospital-data-analysis.csv"

df = pd.read_csv(file_path)

conn = psycopg2.connect(
    dbname="hospital_db",
    user="postgres",
    password="009111",
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