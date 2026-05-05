"""Generador del notebook - Parte 1: Setup, Carga, Sección 3"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.11.0'}
}

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

cells = []

# ── Título ───────────────────────────────────────────────────
cells.append(md("""# 🚦 Taller — Predicción de Accidentalidad
## Aprendizaje Automático · MCDA 2026-1
**Profesores:** Pablo Saldarriaga / Marco Teran  
**Tema:** Predicción de accidentalidad por barrio y hora  
**Tipo:** Clasificación binaria — Clases desbalanceadas — SQLite

---
> **Pregunta:** ¿Cuál es la probabilidad de que ocurra al menos un accidente en el barrio B a la hora h?
"""))

# ── 0. Imports ───────────────────────────────────────────────
cells.append(md("## 0. Importaciones y Configuración"))
cells.append(code("""import subprocess, sys
for pkg in ['imbalanced-learn','xgboost','lightgbm','seaborn']:
    try: __import__(pkg.replace('-','_'))
    except ImportError: subprocess.check_call([sys.executable,'-m','pip','install',pkg,'-q'])
print("✓ Dependencias OK")
"""))

cells.append(code("""import sqlite3, warnings, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve, roc_curve, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.4f}'.format)

plt.rcParams.update({
    'figure.dpi': 120, 'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa', 'axes.grid': True,
    'grid.color': 'white', 'grid.linewidth': 0.8,
    'axes.titlesize': 13, 'axes.labelsize': 11,
})
DIAS = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
MESES = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
print(f"✓ pandas {pd.__version__} | numpy {np.__version__} | xgboost {xgb.__version__}")
"""))

# ── 1. Carga de datos ────────────────────────────────────────
cells.append(md("## 1. Conexión y Carga desde SQLite"))
cells.append(code("""DB_PATH = 'data_accidentes.sqlite3'
assert os.path.exists(DB_PATH), f"No encontrado: {DB_PATH}"

con = sqlite3.connect(DB_PATH)
accidentes = pd.read_sql('SELECT * FROM accidentes',      con, parse_dates=['TW'])
clima       = pd.read_sql('SELECT * FROM clima',          con, parse_dates=['TW'])
raw         = pd.read_sql('SELECT * FROM raw_accidentes', con, parse_dates=['TW'])
con.close()

print("✓ Tablas cargadas")
for nombre, df_ in [('accidentes', accidentes), ('clima', clima), ('raw_accidentes', raw)]:
    print(f"  {nombre:15s}: {df_.shape[0]:>10,} filas × {df_.shape[1]} cols")
"""))

# ── 3. Construcción del dataset ───────────────────────────────
cells.append(md("""---
## 3. Pregunta a Responder — Construcción del Dataset

### 3.1 Unidad de análisis
La **unidad de análisis** es el par **(BARRIO, TW)** donde `TW` es una marca temporal truncada a la hora. Cada fila representa una combinación única barrio × hora y la variable objetivo indica si **al menos un accidente** ocurrió ahí.

### 3.2 Construcción de casos negativos
- `target = 1` → el par (BARRIO, TW) aparece en la tabla `accidentes`  
- `target = 0` → el par (BARRIO, TW) está en `clima` pero **no** en `accidentes`

Se realiza un **LEFT JOIN** desde `clima` (que cubre todas las combinaciones) hacia `accidentes`.
"""))

cells.append(code("""# Normalizar BARRIO
clima_n = clima.copy()
acc_n   = accidentes.copy()
clima_n['BARRIO'] = clima_n['BARRIO'].str.strip().str.lower()
acc_n['BARRIO']   = acc_n['BARRIO'].str.strip().str.lower()
acc_n['target']   = 1

# LEFT JOIN clima ← accidentes
df = clima_n.merge(
    acc_n[['TW','BARRIO','target']],
    on=['TW','BARRIO'], how='left'
)
df['target'] = df['target'].fillna(0).astype(int)

# Componentes temporales desde TW
df['hora']     = df['TW'].dt.hour
df['dia']      = df['TW'].dt.day
df['mes']      = df['TW'].dt.month
df['anio']     = df['TW'].dt.year
df['dia_sem']  = df['TW'].dt.dayofweek   # 0=Lun … 6=Dom
df['dia_anio'] = df['TW'].dt.dayofyear
df['es_finde'] = df['dia_sem'].isin([5,6]).astype(int)

print(f"✓ Dataset construido: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"  Período: {df['TW'].min()} → {df['TW'].max()}")
"""))

cells.append(code("""# ── 3.3 Proporción de la clase positiva ─────────────────────
total      = len(df)
positivos  = df['target'].sum()
negativos  = total - positivos
pct_pos    = positivos / total * 100

print("="*52)
print("  DISTRIBUCIÓN DE CLASES")
print("="*52)
print(f"  Total observaciones : {total:>10,}")
print(f"  Con accidente  (1)  : {positivos:>10,}  ({pct_pos:.2f}%)")
print(f"  Sin accidente  (0)  : {negativos:>10,}  ({100-pct_pos:.2f}%)")
print(f"  Ratio desbalance    : 1 : {negativos//positivos:.0f}")
print()
print("→ El 1.51% de positivos justifica métricas especializadas")
print("  (PR-AUC, F1, Recall) y estrategias de balanceo.")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Pastel
axes[0].pie([positivos, negativos],
    labels=[f'Con accidente\\n{pct_pos:.2f}%', f'Sin accidente\\n{100-pct_pos:.2f}%'],
    colors=['#e74c3c','#2ecc71'], autopct='%1.2f%%', startangle=90,
    wedgeprops={'edgecolor':'white','linewidth':2})
axes[0].set_title('Distribución de la Variable Objetivo', fontweight='bold')

# Barras
axes[1].bar(['Sin accidente (0)','Con accidente (1)'],
    [negativos, positivos], color=['#2ecc71','#e74c3c'], edgecolor='white', linewidth=1.5)
axes[1].set_ylabel('Número de registros')
axes[1].set_title('Clases Desbalanceadas — Escala Real', fontweight='bold')
for i, v in enumerate([negativos, positivos]):
    axes[1].text(i, v*1.01, f'{v:,}', ha='center', fontweight='bold')

plt.suptitle('3.3 — Clase Positiva: 1.51% de los Datos', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

# Guardar cells en archivo temporal
import pickle, os
os.makedirs('nb_parts', exist_ok=True)
with open('nb_parts/cells_p1.pkl', 'wb') as f:
    pickle.dump(cells, f)

print(f"✓ Parte 1 lista: {len(cells)} celdas")
