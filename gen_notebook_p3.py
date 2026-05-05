"""Parte 3: Feature Engineering (4.3)"""
import pickle
with open('nb_parts/cells_p2.pkl','rb') as f: cells = pickle.load(f)
import nbformat as nbf
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

cells.append(md("---\n## 4.3 Ingeniería de Características"))

cells.append(code("""# ── Variables cíclicas con seno/coseno ──────────────────────
df['hora_sin']     = np.sin(2 * np.pi * df['hora'] / 24)
df['hora_cos']     = np.cos(2 * np.pi * df['hora'] / 24)
df['dia_sem_sin']  = np.sin(2 * np.pi * df['dia_sem'] / 7)
df['dia_sem_cos']  = np.cos(2 * np.pi * df['dia_sem'] / 7)
df['dia_anio_sin'] = np.sin(2 * np.pi * df['dia_anio'] / 365)
df['dia_anio_cos'] = np.cos(2 * np.pi * df['dia_anio'] / 365)
df['mes_sin']      = np.sin(2 * np.pi * df['mes'] / 12)
df['mes_cos']      = np.cos(2 * np.pi * df['mes'] / 12)

print("✓ Variables cíclicas creadas (seno/coseno)")
print("  hora, dia_sem, dia_anio, mes → 8 variables nuevas")
"""))

cells.append(code("""# ── Variable histórica: tasa de accidentalidad por barrio ────
# IMPORTANTE: calcular con datos anteriores al período de validación
# para evitar fuga de información temporal.
# Usamos la media global del período de entrenamiento (definido más adelante).

# Por ahora: tasa histórica global por barrio (se recalculará en el pipeline)
tasa_barrio = (df.groupby('BARRIO')['target']
               .mean()
               .reset_index()
               .rename(columns={'target': 'tasa_hist_barrio'}))

df = df.merge(tasa_barrio, on='BARRIO', how='left')
df['tasa_hist_barrio'] = df['tasa_hist_barrio'].fillna(df['tasa_hist_barrio'].median())

# Tasa histórica por barrio × hora del día
tasa_barrio_hora = (df.groupby(['BARRIO','hora'])['target']
                    .mean()
                    .reset_index()
                    .rename(columns={'target':'tasa_barrio_hora'}))
df = df.merge(tasa_barrio_hora, on=['BARRIO','hora'], how='left')
df['tasa_barrio_hora'] = df['tasa_barrio_hora'].fillna(0)

# Tasa histórica por barrio × día de semana
tasa_barrio_dia = (df.groupby(['BARRIO','dia_sem'])['target']
                   .mean()
                   .reset_index()
                   .rename(columns={'target':'tasa_barrio_dia'}))
df = df.merge(tasa_barrio_dia, on=['BARRIO','dia_sem'], how='left')
df['tasa_barrio_dia'] = df['tasa_barrio_dia'].fillna(0)

print("✓ Variables históricas de barrio creadas")
print("  tasa_hist_barrio, tasa_barrio_hora, tasa_barrio_dia")
"""))

cells.append(code("""# ── Codificación de variables categóricas ────────────────────
from sklearn.preprocessing import LabelEncoder

# BARRIO: frecuencia encoding
freq_barrio = df['BARRIO'].value_counts() / len(df)
df['barrio_freq'] = df['BARRIO'].map(freq_barrio)

# summary / icon: frecuencia encoding
freq_summary = df['summary'].value_counts() / len(df)
freq_icon    = df['icon'].value_counts() / len(df)
df['summary_freq'] = df['summary'].map(freq_summary)
df['icon_freq']    = df['icon'].map(freq_icon)

print("✓ Encoding por frecuencia aplicado: BARRIO, summary, icon")
"""))

cells.append(code("""# ── Lista final de features ──────────────────────────────────
FEATURES = [
    # Climáticas
    'temperature','apparentTemperature','dewPoint','humidity',
    'precipIntensity','precipProbability','windSpeed','windBearing',
    'cloudCover','uvIndex','visibility',
    # Temporales cíclicas
    'hora_sin','hora_cos','dia_sem_sin','dia_sem_cos',
    'dia_anio_sin','dia_anio_cos','mes_sin','mes_cos',
    # Temporales lineales
    'hora','dia_sem','mes','anio','es_finde',
    # Históricas por barrio
    'tasa_hist_barrio','tasa_barrio_hora','tasa_barrio_dia',
    # Encoding categóricas
    'barrio_freq','summary_freq','icon_freq',
]
TARGET = 'target'

print(f"✓ Feature set final: {len(FEATURES)} variables")
print("\\nResumen de features por grupo:")
print("  Climáticas          : 11")
print("  Temporales cíclicas : 8")
print("  Temporales lineales : 5")
print("  Históricas barrio   : 3")
print("  Encoding categóricas: 3")
print(f"  TOTAL               : {len(FEATURES)}")
"""))

cells.append(code("""# ── Verificar que no hay nulos en las features finales ────────
nulos_feat = df[FEATURES].isnull().sum()
nulos_feat = nulos_feat[nulos_feat > 0]
if len(nulos_feat) == 0:
    print("✓ Sin nulos en el feature set — listo para modelar")
else:
    print("Columnas con nulos:")
    print(nulos_feat)
    # Imputar con mediana
    for col in nulos_feat.index:
        df[col] = df[col].fillna(df[col].median())
    print("✓ Nulos imputados con mediana")
"""))

with open('nb_parts/cells_p3.pkl','wb') as f: pickle.dump(cells, f)
print(f"✓ Parte 3: {len(cells)} celdas")
