import sqlite3
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

DB = r'c:\Users\salds\OneDrive\Documentos\AprendizajeAutomaticoT1\data_accidentes.sqlite3'
con = sqlite3.connect(DB)

accidentes = pd.read_sql("SELECT * FROM accidentes", con, parse_dates=['TW'])
clima       = pd.read_sql("SELECT * FROM clima", con, parse_dates=['TW'])
raw         = pd.read_sql("SELECT * FROM raw_accidentes", con, parse_dates=['TW'])
con.close()

SEP = "\n" + "="*70 + "\n"

# ─────────────────────────────────────────────────────────────
print(SEP + "1. FORMA DE LAS TABLAS")
for nombre, df in [("accidentes", accidentes), ("clima", clima), ("raw_accidentes", raw)]:
    print(f"  {nombre}: {df.shape[0]:,} filas x {df.shape[1]} columnas")

# ─────────────────────────────────────────────────────────────
print(SEP + "2. COLUMNAS REALES vs ESPERADAS (según PDF)")

print("\n--- accidentes ---")
print(sorted(accidentes.columns.tolist()))
esperado_acc = {'TW','BARRIO','Lat','Lon','Dia_sem','Mes','Dia','Hora'}
faltantes = esperado_acc - set(accidentes.columns)
print(f"  Faltantes vs PDF: {faltantes if faltantes else 'ninguna ✓'}")

print("\n--- clima ---")
print(sorted(clima.columns.tolist()))
esperado_cli = {'TW','BARRIO','summary','icon','precipIntensity','precipProbability',
                'temperature','apparentTemperature','dewPoint','humidity',
                'windSpeed','windBearing','cloudCover','uvIndex','visibility'}
faltantes = esperado_cli - set(clima.columns)
print(f"  Faltantes vs PDF: {faltantes if faltantes else 'ninguna ✓'}")

print("\n--- raw_accidentes ---")
print(sorted(raw.columns.tolist()))
esperado_raw = {'TW','FECHA','HORA','Hora_num','Dia','Mes','MES_NOMBRE','Dia_sem',
                'PERIODO','BARRIO','COMUNA','Lat','Lon','CLASE','GRAVEDAD','DISENO',
                'DIRECCION','OBJECTID','RADICADO'}
faltantes = esperado_raw - set(raw.columns)
print(f"  Faltantes vs PDF: {faltantes if faltantes else 'ninguna ✓'}")

# ─────────────────────────────────────────────────────────────
print(SEP + "3. RANGO TEMPORAL")
for nombre, df in [("accidentes", accidentes), ("clima", clima), ("raw_accidentes", raw)]:
    tw_col = 'TW'
    print(f"  {nombre}: {df[tw_col].min()} → {df[tw_col].max()}")

# ─────────────────────────────────────────────────────────────
print(SEP + "4. BARRIOS: consistencia entre tablas")
barrios_acc  = set(accidentes['BARRIO'].str.strip().str.lower().dropna().unique())
barrios_cli  = set(clima['BARRIO'].str.strip().str.lower().dropna().unique())
barrios_raw  = set(raw['BARRIO'].str.strip().str.lower().dropna().unique())

print(f"  Barrios en accidentes : {len(barrios_acc)}")
print(f"  Barrios en clima      : {len(barrios_cli)}")
print(f"  Barrios en raw        : {len(barrios_raw)}")
print(f"  En accidentes pero NO en clima : {len(barrios_acc - barrios_cli)} → {sorted(list(barrios_acc - barrios_cli))[:10]}")
print(f"  En clima pero NO en accidentes : {len(barrios_cli - barrios_acc)}")
print(f"  En raw pero NO en clima        : {len(barrios_raw - barrios_cli)} → {sorted(list(barrios_raw - barrios_cli))[:10]}")

# ─────────────────────────────────────────────────────────────
print(SEP + "5. FORMATO DE TW: consistencia")
print("  accidentes TW sample:", accidentes['TW'].head(3).tolist())
print("  clima TW sample:", clima['TW'].head(3).tolist())
print("  raw TW sample:", raw['TW'].head(3).tolist())

# Verificar que TW ya está como datetime
print(f"  accidentes TW dtype: {accidentes['TW'].dtype}")
print(f"  clima TW dtype      : {clima['TW'].dtype}")
print(f"  raw TW dtype        : {raw['TW'].dtype}")

# ─────────────────────────────────────────────────────────────
print(SEP + "6. DESBALANCE DE CLASES (estimación)")
total_pares_clima = len(clima)
total_positivos   = len(accidentes)
pct_positivos = total_positivos / total_pares_clima * 100
print(f"  Total pares (barrio,hora) en clima   : {total_pares_clima:,}")
print(f"  Pares con accidente (accidentes)     : {total_positivos:,}")
print(f"  % positivos estimado                 : {pct_positivos:.2f}%")

# ─────────────────────────────────────────────────────────────
print(SEP + "7. DUPLICADOS")
dup_acc = accidentes.duplicated(subset=['TW','BARRIO']).sum()
dup_cli = clima.duplicated(subset=['TW','BARRIO']).sum()
dup_raw = raw.duplicated(subset=['OBJECTID']).sum()
print(f"  Duplicados (TW,BARRIO) en accidentes : {dup_acc}")
print(f"  Duplicados (TW,BARRIO) en clima      : {dup_cli}")
print(f"  Duplicados OBJECTID en raw           : {dup_raw}")

# ─────────────────────────────────────────────────────────────
print(SEP + "8. VALORES NULOS")
for nombre, df in [("accidentes", accidentes), ("clima", clima), ("raw_accidentes", raw)]:
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    if len(nulos) > 0:
        print(f"\n  {nombre}:")
        print(nulos.to_string())
    else:
        print(f"\n  {nombre}: sin nulos ✓")

# ─────────────────────────────────────────────────────────────
print(SEP + "9. HORAS (deben ser 0–23)")
print("  accidentes - Hora:", sorted(accidentes['Hora'].dropna().unique().tolist()))
if 'Hora_num' in raw.columns:
    print("  raw_accidentes - Hora_num:", sorted(raw['Hora_num'].dropna().unique().tolist()))

# ─────────────────────────────────────────────────────────────
print(SEP + "10. CRUCE: accidentes vs clima (¿se pueden joinear?)")
# Muestra la cantidad de (TW,BARRIO) de accidentes que sí existen en clima
acc_norm = accidentes.copy()
cli_norm  = clima.copy()
acc_norm['BARRIO_L'] = acc_norm['BARRIO'].str.strip().str.lower()
cli_norm['BARRIO_L']  = cli_norm['BARRIO'].str.strip().str.lower()
merged = acc_norm.merge(cli_norm[['TW','BARRIO_L']], left_on=['TW','BARRIO_L'], right_on=['TW','BARRIO_L'], how='left', indicator=True)
match = (merged['_merge'] == 'both').sum()
print(f"  Positivos que tienen contraparte en clima: {match:,} de {len(accidentes):,} ({match/len(accidentes)*100:.1f}%)")

print(SEP + "ANÁLISIS COMPLETADO ✓")
