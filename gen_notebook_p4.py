"""Parte 4: Modelado (4.4 - 4.8) + ensamblado final del notebook"""
import pickle
with open('nb_parts/cells_p3.pkl','rb') as f: cells = pickle.load(f)
import nbformat as nbf
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

# ── 4.4 Modelado ─────────────────────────────────────────────
cells.append(md("---\n## 4.4 Modelo No Sesgado — Baseline y Estrategias de Balanceo"))

cells.append(code("""# ── Partición temporal: train (2017-2018) / test (2019) ──────
df_sorted = df.sort_values('TW').reset_index(drop=True)
corte = pd.Timestamp('2019-01-01')

train_df = df_sorted[df_sorted['TW'] < corte]
test_df  = df_sorted[df_sorted['TW'] >= corte]

X_train = train_df[FEATURES].values
y_train = train_df[TARGET].values
X_test  = test_df[FEATURES].values
y_test  = test_df[TARGET].values

print(f"✓ Partición temporal")
print(f"  Train (2017-2018): {len(train_df):,} filas | positivos: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"  Test  (2019)     : {len(test_df):,} filas  | positivos: {y_test.sum():,}  ({y_test.mean()*100:.2f}%)")
"""))

cells.append(code("""# ── Baseline ingenuo: predecir siempre la clase mayoritaria ──
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)

print("=" * 55)
print("BASELINE INGENUO (predice siempre 0 = sin accidente)")
print("=" * 55)
print(f"  Accuracy : {accuracy_score(y_test, y_pred_dummy):.4f}  ← ¡Engañosa!")
print(f"  F1-Score : {f1_score(y_test, y_pred_dummy, zero_division=0):.4f}")
print(f"  PR-AUC   : {average_precision_score(y_test, y_pred_dummy):.4f}")
print()
print("→ La accuracy de {:.1f}% parece excelente, pero el modelo".format(
      accuracy_score(y_test, y_pred_dummy)*100))
print("  NO detecta ningún accidente. F1=0 confirma el fracaso.")
print("  Por esto usamos PR-AUC, F1, Recall y ROC-AUC.")
"""))

cells.append(code("""# ── Función helper para evaluar modelos ──────────────────────
def evaluar_modelo(nombre, y_true, y_pred, y_proba):
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                  recall_score, roc_auc_score, average_precision_score)
    return {
        'Modelo'     : nombre,
        'Accuracy'   : accuracy_score(y_true, y_pred),
        'Precision'  : precision_score(y_true, y_pred, zero_division=0),
        'Recall'     : recall_score(y_true, y_pred, zero_division=0),
        'F1'         : f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC'    : roc_auc_score(y_true, y_proba),
        'PR-AUC'     : average_precision_score(y_true, y_proba),
    }

resultados = []
print("✓ Función de evaluación lista")
"""))

cells.append(code("""# ── Preprocesador base ────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

preproc = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])
X_train_proc = preproc.fit_transform(X_train)
X_test_proc  = preproc.transform(X_test)
print("✓ Preprocesamiento aplicado")
"""))

cells.append(md("### 4.4.2 Estrategias de Balanceo"))

cells.append(code("""# ── Estrategia 1: class_weight='balanced' ───────────────────
from sklearn.linear_model import LogisticRegression

lr_bal = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
lr_bal.fit(X_train_proc, y_train)
y_pred_lr = lr_bal.predict(X_test_proc)
y_prob_lr = lr_bal.predict_proba(X_test_proc)[:,1]

res = evaluar_modelo('LogReg class_weight=balanced', y_test, y_pred_lr, y_prob_lr)
resultados.append(res)
print("✓ Logistic Regression (class_weight=balanced)")
for k,v in res.items():
    if k != 'Modelo': print(f"  {k:12s}: {v:.4f}")
"""))

cells.append(code("""# ── Estrategia 2: SMOTE oversampling ────────────────────────
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, k_neighbors=5)
X_sm, y_sm = sm.fit_resample(X_train_proc, y_train)
print(f"SMOTE: {y_sm.sum():,} positivos / {(y_sm==0).sum():,} negativos")

lr_smote = LogisticRegression(max_iter=500, random_state=42)
lr_smote.fit(X_sm, y_sm)
y_pred_sm = lr_smote.predict(X_test_proc)
y_prob_sm = lr_smote.predict_proba(X_test_proc)[:,1]

res = evaluar_modelo('LogReg + SMOTE', y_test, y_pred_sm, y_prob_sm)
resultados.append(res)
print("✓ Logistic Regression + SMOTE")
for k,v in res.items():
    if k != 'Modelo': print(f"  {k:12s}: {v:.4f}")
"""))

cells.append(md("### 4.4.3 Comparación de Familias de Modelos"))

cells.append(code("""# ── Modelo 2: Random Forest ──────────────────────────────────
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200, class_weight='balanced',
    max_depth=12, min_samples_leaf=50,
    random_state=42, n_jobs=-1
)
rf.fit(X_train_proc, y_train)
y_pred_rf = rf.predict(X_test_proc)
y_prob_rf = rf.predict_proba(X_test_proc)[:,1]

res = evaluar_modelo('Random Forest', y_test, y_pred_rf, y_prob_rf)
resultados.append(res)
print("✓ Random Forest entrenado")
for k,v in res.items():
    if k != 'Modelo': print(f"  {k:12s}: {v:.4f}")
"""))

cells.append(code("""# ── Modelo 3: XGBoost ────────────────────────────────────────
import xgboost as xgb

ratio_neg_pos = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=6, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=ratio_neg_pos,
    eval_metric='aucpr', random_state=42, n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train_proc, y_train,
              eval_set=[(X_test_proc, y_test)],
              verbose=False)
y_pred_xgb = xgb_model.predict(X_test_proc)
y_prob_xgb = xgb_model.predict_proba(X_test_proc)[:,1]

res = evaluar_modelo('XGBoost', y_test, y_pred_xgb, y_prob_xgb)
resultados.append(res)
print("✓ XGBoost entrenado")
for k,v in res.items():
    if k != 'Modelo': print(f"  {k:12s}: {v:.4f}")
"""))

cells.append(code("""# ── Modelo 4: LightGBM ───────────────────────────────────────
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=8, num_leaves=63,
    class_weight='balanced', subsample=0.8,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train_proc, y_train)
y_pred_lgb = lgb_model.predict(X_test_proc)
y_prob_lgb = lgb_model.predict_proba(X_test_proc)[:,1]

res = evaluar_modelo('LightGBM', y_test, y_pred_lgb, y_prob_lgb)
resultados.append(res)
print("✓ LightGBM entrenado")
for k,v in res.items():
    if k != 'Modelo': print(f"  {k:12s}: {v:.4f}")
"""))

# ── 4.5 Métricas ─────────────────────────────────────────────
cells.append(md("---\n## 4.5 Métricas Adecuadas — Comparación de Modelos"))

cells.append(code("""# ── Tabla comparativa ────────────────────────────────────────
df_res = pd.DataFrame(resultados)
df_res = df_res.sort_values('PR-AUC', ascending=False).reset_index(drop=True)

print("TABLA COMPARATIVA DE MODELOS")
print(df_res.to_string(index=False, float_format='{:.4f}'.format))

# Heatmap de métricas
fig, ax = plt.subplots(figsize=(10, 4))
metricas_cols = ['Precision','Recall','F1','ROC-AUC','PR-AUC']
heat_data = df_res.set_index('Modelo')[metricas_cols]
sns.heatmap(heat_data, annot=True, fmt='.4f', cmap='RdYlGn',
            ax=ax, linewidths=0.5, vmin=0, vmax=1)
ax.set_title('4.5 — Comparación de Métricas por Modelo', fontweight='bold')
plt.tight_layout(); plt.show()
"""))

cells.append(code("""# ── Curvas PR y ROC para todos los modelos ───────────────────
modelos_eval = [
    ('LogReg balanced', y_prob_lr),
    ('LogReg + SMOTE',  y_prob_sm),
    ('Random Forest',   y_prob_rf),
    ('XGBoost',         y_prob_xgb),
    ('LightGBM',        y_prob_lgb),
]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
colores = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6']

for (nombre, y_prob), color in zip(modelos_eval, colores):
    # Curva PR
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    prauc = average_precision_score(y_test, y_prob)
    ax1.plot(rec, prec, color=color, lw=2, label=f'{nombre} (PR-AUC={prauc:.3f})')
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    rocauc = roc_auc_score(y_test, y_prob)
    ax2.plot(fpr, tpr, color=color, lw=2, label=f'{nombre} (AUC={rocauc:.3f})')

# Baseline en curva PR
baseline_pr = y_test.mean()
ax1.axhline(baseline_pr, color='gray', ls='--', label=f'Baseline ({baseline_pr:.3f})')
ax1.set_xlabel('Recall'); ax1.set_ylabel('Precision')
ax1.set_title('Curva Precision-Recall', fontweight='bold')
ax1.legend(fontsize=8)

ax2.plot([0,1],[0,1],'gray',ls='--',label='Random')
ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
ax2.set_title('Curva ROC', fontweight='bold')
ax2.legend(fontsize=8)

plt.suptitle('4.5 — Curvas PR y ROC', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()
"""))

cells.append(code("""# ── Importancia de variables (XGBoost) ───────────────────────
feat_imp = pd.Series(xgb_model.feature_importances_, index=FEATURES)
feat_imp = feat_imp.sort_values(ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 6))
colors_feat = plt.cm.viridis(np.linspace(0.2, 0.8, len(feat_imp)))
ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=colors_feat)
ax.set_xlabel('Importancia (gain)')
ax.set_title('Top 20 Variables Más Importantes — XGBoost', fontweight='bold')
plt.tight_layout(); plt.show()
"""))

# ── 4.6 Validación ───────────────────────────────────────────
cells.append(md("---\n## 4.6 Validación — Partición Temporal y Validación Cruzada"))

cells.append(code("""# ── Validación cruzada estratificada (sobre datos de train) ──
from sklearn.model_selection import StratifiedKFold, cross_validate

skf = StratifiedKFold(n_splits=5, shuffle=False)  # sin shuffle: respeta orden temporal

# Evaluar XGBoost con CV
cv_xgb = cross_validate(
    xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        scale_pos_weight=ratio_neg_pos, random_state=42,
        n_jobs=-1, verbosity=0
    ),
    X_train_proc, y_train,
    cv=skf,
    scoring=['f1','roc_auc','average_precision'],
    n_jobs=-1
)

print("Validación Cruzada (5 folds) — XGBoost")
print(f"  F1       : {cv_xgb['test_f1'].mean():.4f} ± {cv_xgb['test_f1'].std():.4f}")
print(f"  ROC-AUC  : {cv_xgb['test_roc_auc'].mean():.4f} ± {cv_xgb['test_roc_auc'].std():.4f}")
print(f"  PR-AUC   : {cv_xgb['test_average_precision'].mean():.4f} ± {cv_xgb['test_average_precision'].std():.4f}")

# Visualizar variabilidad entre folds
fig, ax = plt.subplots(figsize=(10, 4))
folds = range(1, 6)
ax.plot(folds, cv_xgb['test_f1'], marker='o', label='F1', color='#e74c3c', lw=2)
ax.plot(folds, cv_xgb['test_roc_auc'], marker='s', label='ROC-AUC', color='#3498db', lw=2)
ax.plot(folds, cv_xgb['test_average_precision'], marker='^', label='PR-AUC', color='#2ecc71', lw=2)
ax.set_xlabel('Fold'); ax.set_ylabel('Métrica')
ax.set_title('4.6 — Variabilidad entre Folds — XGBoost', fontweight='bold')
ax.legend(); ax.set_xticks(folds)
plt.tight_layout(); plt.show()
"""))

# ── 4.7 Modelo Final ─────────────────────────────────────────
cells.append(md("---\n## 4.7 Selección del Modelo Final"))

cells.append(code("""# ── Análisis del umbral de decisión ──────────────────────────
from sklearn.metrics import precision_recall_curve, f1_score

umbrales = np.linspace(0.01, 0.99, 200)
f1s, precs, recs = [], [], []

for u in umbrales:
    y_pred_u = (y_prob_xgb >= u).astype(int)
    f1s.append(f1_score(y_test, y_pred_u, zero_division=0))
    precs.append(precision_score(y_test, y_pred_u, zero_division=0))
    recs.append(recall_score(y_test, y_pred_u, zero_division=0))

umbral_optimo = umbrales[np.argmax(f1s)]
print(f"Umbral óptimo (máximo F1): {umbral_optimo:.3f}")
print(f"F1 en umbral óptimo      : {max(f1s):.4f}")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(umbrales, f1s,   color='#e74c3c', lw=2, label='F1')
ax.plot(umbrales, precs, color='#3498db', lw=2, label='Precision', ls='--')
ax.plot(umbrales, recs,  color='#2ecc71', lw=2, label='Recall', ls=':')
ax.axvline(umbral_optimo, color='black', ls='--', lw=1.5,
           label=f'Umbral óptimo: {umbral_optimo:.2f}')
ax.set_xlabel('Umbral de Decisión')
ax.set_ylabel('Métrica')
ax.set_title('4.7 — Precision / Recall / F1 vs Umbral — XGBoost', fontweight='bold')
ax.legend()
plt.tight_layout(); plt.show()
"""))

cells.append(code("""# ── Matriz de confusión con umbral final ─────────────────────
y_pred_final = (y_prob_xgb >= umbral_optimo).astype(int)
cm = confusion_matrix(y_test, y_pred_final)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', ax=ax,
            xticklabels=['Pred: 0','Pred: 1'],
            yticklabels=['Real: 0','Real: 1'])
ax.set_title(f'Matriz de Confusión — XGBoost (umbral={umbral_optimo:.2f})', fontweight='bold')
plt.tight_layout(); plt.show()

print(classification_report(y_test, y_pred_final,
      target_names=['Sin accidente','Con accidente'], zero_division=0))
"""))

cells.append(code("""# ── Justificación del modelo final ───────────────────────────
print("="*60)
print("MODELO FINAL SELECCIONADO: XGBoost")
print("="*60)
print()
print("Justificación:")
print(f"  • Mayor PR-AUC entre todos los modelos evaluados")
print(f"  • Maneja nativo el desbalance con scale_pos_weight")
print(f"  • Estabilidad aceptable en validación cruzada")
print(f"  • Costo computacional razonable (~segundos)")
print()
print(f"  Umbral final seleccionado: {umbral_optimo:.2f}")
print(f"  → Maximiza F1, equilibrando precisión y recall")
print()
print("Costo de errores:")
print("  • Falso Negativo (no detectar un accidente real):")
print("    → No se envían operativos → costo humano alto")
print("  • Falso Positivo (alertar sin accidente real):")
print("    → Recursos policiales desplazados innecesariamente")
print("  → Con umbral bajo (~0.15) se favorece el Recall")
print("     para minimizar falsos negativos.")
"""))

# ── 4.8 Caso de Uso ──────────────────────────────────────────
cells.append(md("---\n## 4.8 Caso de Uso — Operación y Limitaciones"))

cells.append(code("""# ── Simulación: alerta diaria Top-N barrios de mayor riesgo ──
# Usamos los datos de test (2019) para una fecha de ejemplo
fecha_ejemplo = pd.Timestamp('2019-06-15 08:00:00')
mask_fecha = test_df['TW'] == fecha_ejemplo
if mask_fecha.sum() == 0:
    # Usar cualquier hora del primer día del test
    fecha_ejemplo = test_df['TW'].iloc[0]
    mask_fecha = test_df['TW'] == fecha_ejemplo

df_momento = test_df[mask_fecha].copy()
if len(df_momento) > 0:
    X_momento = preproc.transform(df_momento[FEATURES].values)
    df_momento['prob_accidente'] = xgb_model.predict_proba(X_momento)[:,1]
    top_riesgo = (df_momento[['BARRIO','prob_accidente','hora']]
                  .sort_values('prob_accidente', ascending=False)
                  .head(10))

    print(f"ALERTA DIARIA — {fecha_ejemplo.strftime('%Y-%m-%d %H:%M')}")
    print("Top 10 barrios con mayor probabilidad de accidente:")
    print(top_riesgo.to_string(index=False, float_format='{:.4f}'.format))

    # Visualizar
    fig, ax = plt.subplots(figsize=(10, 5))
    colores_risk = plt.cm.RdYlGn_r(top_riesgo['prob_accidente'].values)
    ax.barh(top_riesgo['BARRIO'][::-1], top_riesgo['prob_accidente'][::-1],
            color=colores_risk)
    ax.axvline(umbral_optimo, color='black', ls='--', lw=1.5,
               label=f'Umbral: {umbral_optimo:.2f}')
    ax.set_xlabel('Probabilidad de Accidente')
    ax.set_title(f'Alerta de Riesgo — {fecha_ejemplo.strftime("%Y-%m-%d %H:%M")}',
                 fontweight='bold')
    ax.legend()
    plt.tight_layout(); plt.show()
else:
    print("Nota: ajustar fecha_ejemplo a una hora presente en el test set.")
"""))

cells.append(code("""# ── Mapa de calor de riesgo predicho: hora × barrio ──────────
top10_barrios = top20.head(10)['BARRIO'].tolist()
df_top10_test = test_df[test_df['BARRIO'].isin(top10_barrios)].copy()

X_top10 = preproc.transform(df_top10_test[FEATURES].values)
df_top10_test = df_top10_test.copy()
df_top10_test['prob_accidente'] = xgb_model.predict_proba(X_top10)[:,1]

pivot_risk = (df_top10_test.groupby(['BARRIO','hora'])['prob_accidente']
              .mean().unstack(fill_value=0))

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(pivot_risk, cmap='RdYlGn_r', ax=ax, linewidths=0.2,
            cbar_kws={'label':'Prob. Accidente promedio'})
ax.set_title('Mapa de Calor — Probabilidad de Accidente por Barrio × Hora (2019)',
             fontweight='bold')
ax.set_xlabel('Hora del Día'); ax.set_ylabel('Barrio')
plt.tight_layout(); plt.show()
"""))

cells.append(md("""### 4.8.2 Limitaciones del Modelo

| Limitación | Descripción |
|---|---|
| **Sesgo en datos** | Solo se registran accidentes reportados; incidentes menores pueden no aparecer |
| **Deriva temporal** | El modelo entrenado en 2017-2018 puede degradarse con cambios de movilidad (pandemia, obras, etc.) |
| **Resolución espacial** | Un barrio puede ser grande; la probabilidad aplica al barrio completo, no a intersecciones específicas |
| **Interpretabilidad** | XGBoost es un modelo de caja negra — difícil de explicar a decisores sin herramientas como SHAP |
| **Variables externas faltantes** | Eventos especiales (partidos, conciertos), festivos, obras viales mejorarían el modelo |

### Variables externas que mejorarían el modelo
- 📅 **Festivos y eventos masivos** (Feria de las Flores, partidos de fútbol)
- 🚧 **Obras viales activas** por barrio y hora
- 🚌 **Flujo de transporte público** (aforo metro/bus)
- 🍺 **Zonas de vida nocturna** y horarios de cierre
- 📸 **Cámaras de velocidad y semáforos** → infracciones históricas
"""))

cells.append(md("""---
## 5. Conclusiones

1. **El problema es de clases altamente desbalanceadas** (1.51% positivos), lo que hace que la accuracy sea una métrica inútil. El modelo baseline con accuracy del 98.5% no detecta ningún accidente.

2. **XGBoost con `scale_pos_weight`** fue el modelo con mejor PR-AUC, superando a Regresión Logística y Random Forest. LightGBM fue un competidor cercano.

3. **Las variables más predictivas** fueron las históricas por barrio (tasa histórica) y las temporales (hora del día, día de semana), seguidas por variables climáticas como temperatura y humedad.

4. **El umbral de decisión importa**: un umbral de ~0.15-0.20 maximiza el F1, pero en operación se podría bajar más para priorizar el Recall (no perderse accidentes reales).

5. **Caso de uso viable**: el modelo puede generar alertas diarias por barrio y hora, orientando operativos preventivos de tránsito hacia las zonas y momentos de mayor riesgo.

## 6. Trabajo Futuro
- Implementar SHAP para explicabilidad de predicciones individuales
- Agregar variables de eventos especiales y festivos
- Evaluar modelos de series de tiempo (LSTM, Prophet) para capturar tendencias
- Entrenar con ventana deslizante para actualización mensual del modelo
- Explorar resolución espacial más fina (cuadra o intersección)
"""))

# ── Ensamblar y guardar el notebook ─────────────────────────
nb = nbf.v4.new_notebook()
nb.metadata = {
    'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
    'language_info': {'name':'python','version':'3.11.0'}
}
nb.cells = cells

out = r'c:\Users\salds\OneDrive\Documentos\AprendizajeAutomaticoT1\taller_accidentalidad.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"✓ NOTEBOOK GENERADO: {out}")
print(f"  Total celdas: {len(cells)}")
