import psycopg2
import streamlit as st

# --- DATABASE CONFIGURATION ---
# Replace with your actual database credentials
DB_HOST = "localhost"
DB_NAME = "plant"
DB_USER = "postgres"
DB_PASS = "123"
DB_PORT = 5432

# --- DATABASE CONNECTION ---
# @st.cache_resource
def get_db_connection():
    """Establishes a connection to the database."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"Error connecting to the database: {e}")
        return None

# --- DATA FETCHING LOGIC ---
def get_plant_info(disease_name):
    """Fetches information about a specific plant disease from the database."""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                # Query uses the disease name directly, assuming 'Apple Tree' is implicit
                cur.execute("SELECT description, treatment FROM diseases WHERE class_name = %s", (disease_name,))
                plant_data = cur.fetchone()
                return plant_data
        except psycopg2.Error as e:
            st.error(f"Error querying the database: {e}")
        finally:
            conn.close()
    return None

get_plant_info("Apple___Apple_scab")