import sqlite3
import pandas as pd

con = sqlite3.connect(r'c:\Users\salds\OneDrive\Documentos\AprendizajeAutomaticoT1\data_accidentes.sqlite3')
cursor = con.cursor()

# List tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print("TABLAS DISPONIBLES:", tables)

for tname in tables:
    count_row = cursor.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()
    print(f"\n=== {tname} === ({count_row[0]} filas)")
    df = pd.read_sql(f"SELECT * FROM {tname} LIMIT 3", con)
    print("Columnas y tipos:")
    print(df.dtypes)
    print("Primeras 3 filas:")
    print(df.to_string())

con.close()
