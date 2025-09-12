import pandas as pd
from pathlib import Path
import os
from sqlalchemy import create_engine, MetaData, Table, select, insert, update
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
metadata = MetaData()
metadata.reflect(bind=engine)
patient_records = metadata.tables['patient_records']

def load_data_from_db():
    try:
        with engine.connect() as conn:
            df = pd.read_sql(select(patient_records), conn)
            return df
    except SQLAlchemyError as e:
        print(f"Error loading data from DB: {e}")
        return pd.DataFrame()

def save_data_to_db(df):
    """
    Save the DataFrame to the PostgreSQL 'patient_records' table.
    It replaces all rows (like overwriting the CSV).
    """
    try:
        df.to_sql('patient_records', con=engine, if_exists='append', index=False)
        print("Data saved to PostgreSQL successfully.")
    except SQLAlchemyError as e:
        print(f"Error saving data to PostgreSQL: {e}")

# -----------------------------------
# Add a new patient record
# -----------------------------------
def add_patient_record(patient_data: dict):
    required_fields = ['Age', 'Gender', 'Condition', 'Procedure', 'Cost',
                       'Length_of_Stay', 'Readmission', 'Outcome', 'Satisfaction']
    if not all(field in patient_data for field in required_fields):
        # print("DEBUG: patient_data keys received ->", )
        return {"status": "error", "message": f"Missing required fields. Missing: {list(patient_data.keys())}"}

    try:
        with engine.begin() as conn:
            result = conn.execute(
                insert(patient_records).returning(patient_records.c.Patient_ID),
                patient_data
            )
            new_id = result.scalar()
            return {"status": "success", "message": "Patient record added successfully", "patient_id": new_id}
    except SQLAlchemyError as e:
        return {"status": "error", "message": f"Database insert failed: {e}"}

# -----------------------------------
# Update an existing patient record
# -----------------------------------
def update_patient_record(patient_id: int, updates: dict):
    if not patient_id:
        return {"status": "error", "message": "Patient ID is required for update."}
    if not updates:
        return {"status": "error", "message": "No updates provided."}

    try:
        with engine.begin() as conn:
            stmt = (
                update(patient_records)
                .where(patient_records.c.Patient_ID == patient_id)
                .values(**updates)
            )
            result = conn.execute(stmt)

            if result.rowcount == 0:
                return {"status": "error", "message": f"No record found with Patient_ID {patient_id}"}

            return {"status": "success", "message": f"Patient record {patient_id} updated successfully"}
    except SQLAlchemyError as e:
        return {"status": "error", "message": f"Database update failed: {e}"}

# -----------------------------------
# Get patient record(s)
# -----------------------------------
def get_patient_record(patient_id: Optional[int] = None, query_criteria: Optional[dict] = None):
    if not patient_id and not query_criteria:
        return {"status": "error", "message": "Either patient_id or query_criteria must be provided."}

    try:
        with engine.connect() as conn:
            stmt = select(patient_records)

            if patient_id:
                stmt = stmt.where(patient_records.c.Patient_ID == patient_id)
            elif query_criteria:
                for col, val in query_criteria.items():
                    if hasattr(patient_records.c, col):
                        stmt = stmt.where(getattr(patient_records.c, col) == val)
                    else:
                        return {"status": "error", "message": f"Invalid query field '{col}'"}

            result = conn.execute(stmt)
            records = result.mappings().all()

            if not records:
                return {"status": "no_records", "message": "No patient records found matching the criteria."}

            return {"status": "success", "records": records}
    except SQLAlchemyError as e:
        return {"status": "error", "message": f"Database query failed: {e}"}