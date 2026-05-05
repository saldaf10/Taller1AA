"""Generador - Parte 2: EDA (4.1) y Calidad de Datos (4.2)"""
import pickle, os

with open('nb_parts/cells_p1.pkl','rb') as f:
    cells = pickle.load(f)

import nbformat as nbf
def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

DIAS  = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
MESES = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']

# ── 4.1 EDA ──────────────────────────────────────────────────
cells.append(md("---\n## 4.1 Análisis Exploratorio de Datos (EDA)"))

cells.append(md("### 4.1.1 Distribución Temporal"))

cells.append(code("""# Accidentes por hora / día / mes
acc_hora = df[df['target']==1].groupby('hora')['target'].count().reset_index()
acc_hora.columns = ['hora','n']

acc_dia = df[df['target']==1].groupby('dia_sem')['target'].count().reset_index()
acc_dia.columns = ['dia_sem','n']
acc_dia['nombre'] = acc_dia['dia_sem'].map(dict(enumerate(DIAS)))

acc_mes = df[df['target']==1].groupby('mes')['target'].count().reset_index()
acc_mes.columns = ['mes','n']
acc_mes['nombre'] = acc_mes['mes'].map(dict(enumerate(MESES, 1)))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('4.1.1 — Distribución Temporal de Accidentes (2017-2019)',
             fontsize=14, fontweight='bold')

# Hora
pico_hora = acc_hora.loc[acc_hora['n'].idxmax(), 'hora']
axes[0].bar(acc_hora['hora'], acc_hora['n'], color='#e74c3c', edgecolor='white', linewidth=0.5)
axes[0].axvline(pico_hora, color='#2c3e50', ls='--', lw=1.5, label=f'Pico: {pico_hora}h')
axes[0].set_xlabel('Hora del Día'); axes[0].set_ylabel('N° Accidentes')
axes[0].set_title('Por Hora del Día'); axes[0].set_xticks(range(0,24,2))
axes[0].legend()

# Día semana
cols_dia = ['#3498db']*5 + ['#e74c3c']*2
axes[1].bar(acc_dia['nombre'], acc_dia['n'], color=cols_dia, edgecolor='white')
axes[1].set_xlabel('Día'); axes[1].set_title('Por Día de la Semana')
axes[1].tick_params(axis='x', rotation=30)
p1 = mpatches.Patch(color='#3498db', label='Semana')
p2 = mpatches.Patch(color='#e74c3c', label='Fin de semana')
axes[1].legend(handles=[p1,p2])

# Mes
axes[2].bar(acc_mes['nombre'], acc_mes['n'], color='#9b59b6', edgecolor='white')
axes[2].set_xlabel('Mes'); axes[2].set_title('Por Mes')
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout(); plt.show()

pico_dia = acc_dia.loc[acc_dia['n'].idxmax(),'nombre']
pico_mes = acc_mes.loc[acc_mes['n'].idxmax(),'nombre']
print(f"🔍 Hora pico   : {pico_hora}:00")
print(f"🔍 Día pico    : {pico_dia}")
print(f"🔍 Mes pico    : {pico_mes}")
"""))

cells.append(code("""# Heatmap hora × día de semana
pivot = (df[df['target']==1]
         .groupby(['dia_sem','hora'])['target'].count()
         .unstack(fill_value=0))
pivot.index = DIAS

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.2,
            cbar_kws={'label':'N° Accidentes'})
ax.set_title('Mapa de Calor — Accidentes por Día × Hora', fontweight='bold')
ax.set_xlabel('Hora del Día'); ax.set_ylabel('')
plt.tight_layout(); plt.show()
"""))

cells.append(code("""# Tasa real de accidentalidad por hora (positivos / total observaciones esa hora)
tasa = df.groupby('hora')['target'].agg(['sum','count']).reset_index()
tasa.columns = ['hora','acc','total']
tasa['tasa_pct'] = tasa['acc'] / tasa['total'] * 100

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.bar(tasa['hora'], tasa['acc'], color='#3498db', alpha=0.6, label='Accidentes')
ax1.set_xlabel('Hora'); ax1.set_ylabel('N° Accidentes', color='#3498db')
ax1.tick_params(axis='y', labelcolor='#3498db')
ax2 = ax1.twinx()
ax2.plot(tasa['hora'], tasa['tasa_pct'], color='#e74c3c', marker='o', lw=2, label='Tasa %')
ax2.set_ylabel('Tasa de Accidentalidad (%)', color='#e74c3c')
ax2.tick_params(axis='y', labelcolor='#e74c3c')
ax1.set_title('Volumen vs Tasa Real de Accidentalidad por Hora', fontweight='bold')
ax1.set_xticks(range(24))
l1,lb1 = ax1.get_legend_handles_labels()
l2,lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc='upper left')
plt.tight_layout(); plt.show()
print("🔍 La tasa refleja la probabilidad real por hora — más informativa que el volumen bruto.")
"""))

cells.append(md("### 4.1.2 Distribución Espacial"))

cells.append(code("""# Top 20 barrios con más accidentes
top20 = (df[df['target']==1]
         .groupby('BARRIO')['target'].count()
         .sort_values(ascending=False).head(20)
         .reset_index())
top20.columns = ['BARRIO','n']

fig, ax = plt.subplots(figsize=(12, 7))
colores = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(top20)))
bars = ax.barh(top20['BARRIO'][::-1], top20['n'][::-1], color=colores)
ax.set_xlabel('N° Accidentes (2017–2019)')
ax.set_title('Top 20 Barrios — Mayor Accidentalidad', fontweight='bold')
for bar in bars:
    w = bar.get_width()
    ax.text(w+10, bar.get_y()+bar.get_height()/2, f'{int(w):,}', va='center', fontsize=9)
plt.tight_layout(); plt.show()

print("\\n🔍 Top 5 barrios:")
for _, r in top20.head(5).iterrows():
    print(f"  {r['BARRIO']:30s}: {r['n']:,} accidentes")
"""))

cells.append(code("""# Barrios: accidentes por hora del día (top 5 barrios)
top5 = top20.head(5)['BARRIO'].tolist()
df_top5 = df[(df['BARRIO'].isin(top5)) & (df['target']==1)]
pivot_b = df_top5.groupby(['BARRIO','hora'])['target'].count().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 5))
for b in top5:
    if b in pivot_b.index:
        ax.plot(pivot_b.columns, pivot_b.loc[b], marker='o', lw=2, label=b)
ax.set_xlabel('Hora del Día'); ax.set_ylabel('N° Accidentes')
ax.set_title('Perfil Horario de Accidentalidad — Top 5 Barrios', fontweight='bold')
ax.set_xticks(range(24)); ax.legend()
plt.tight_layout(); plt.show()
"""))

cells.append(md("### 4.1.3 Caracterización del Clima y Correlación con Accidentalidad"))

cells.append(code("""# Distribución de variables climáticas: accidente vs no accidente
vars_clima = ['temperature','humidity','precipIntensity','precipProbability',
              'windSpeed','visibility','cloudCover','uvIndex']

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()
fig.suptitle('Variables Climáticas: Con vs Sin Accidente', fontsize=14, fontweight='bold')

sample_neg = df[df['target']==0].sample(n=min(200000,len(df[df['target']==0])),
                                         random_state=42)
df_plot = pd.concat([df[df['target']==1], sample_neg])

for i, var in enumerate(vars_clima):
    ax = axes[i]
    for t, color, label in [(1,'#e74c3c','Con accidente'),(0,'#2ecc71','Sin accidente')]:
        vals = df_plot[df_plot['target']==t][var].dropna()
        ax.hist(vals, bins=40, color=color, alpha=0.5, label=label, density=True)
    ax.set_title(var); ax.set_xlabel(''); ax.legend(fontsize=8)

plt.tight_layout(); plt.show()
"""))

cells.append(code("""# Correlación de variables numéricas con el target
vars_num = ['temperature','apparentTemperature','dewPoint','humidity',
            'precipIntensity','precipProbability','windSpeed','windBearing',
            'cloudCover','uvIndex','visibility','hora','dia_sem','es_finde','mes']

corr = df[vars_num + ['target']].corr()['target'].drop('target').sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if v > 0 else '#3498db' for v in corr.values]
ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1], edgecolor='white')
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Correlación de Pearson con target')
ax.set_title('Correlación de Variables con Accidentalidad', fontweight='bold')
plt.tight_layout(); plt.show()

print("🔍 Variables con mayor correlación (absoluta):")
print(corr.abs().sort_values(ascending=False).head(8).to_string())
"""))

cells.append(code("""# Temperatura promedio: accidente vs no accidente por hora
temp_hora = df.groupby(['hora','target'])['temperature'].mean().unstack()
temp_hora.columns = ['Sin accidente','Con accidente']

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(temp_hora.index, temp_hora['Con accidente'], color='#e74c3c',
        marker='o', lw=2, label='Con accidente')
ax.plot(temp_hora.index, temp_hora['Sin accidente'], color='#2ecc71',
        marker='s', lw=2, label='Sin accidente', ls='--')
ax.set_xlabel('Hora del Día'); ax.set_ylabel('Temperatura Promedio (°C)')
ax.set_title('Temperatura Promedio por Hora — Con vs Sin Accidente', fontweight='bold')
ax.set_xticks(range(24)); ax.legend()
plt.tight_layout(); plt.show()
"""))

# ── 4.2 Calidad ───────────────────────────────────────────────
cells.append(md("---\n## 4.2 Calidad de Datos"))

cells.append(code("""# ── Nulos en el dataset de modelado ─────────────────────────
nulos = df.isnull().sum()
nulos_pct = (nulos / len(df) * 100).round(2)
reporte_nulos = pd.DataFrame({'Nulos': nulos, 'Porcentaje_%': nulos_pct})
reporte_nulos = reporte_nulos[reporte_nulos['Nulos'] > 0].sort_values('Nulos', ascending=False)

print("Columnas con valores nulos:")
print(reporte_nulos.to_string())

fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(reporte_nulos.index[::-1], reporte_nulos['Porcentaje_%'][::-1],
        color='#e74c3c', edgecolor='white')
ax.set_xlabel('% Nulos')
ax.set_title('Valores Nulos por Columna — Dataset Modelado', fontweight='bold')
plt.tight_layout(); plt.show()
"""))

cells.append(code("""# ── Duplicados ───────────────────────────────────────────────
dup = df.duplicated(subset=['TW','BARRIO']).sum()
print(f"Duplicados (TW, BARRIO): {dup}")

# ── Consistencia de barrios entre tablas ─────────────────────
barrios_acc  = set(accidentes['BARRIO'].str.strip().str.lower().dropna().unique())
barrios_cli  = set(clima['BARRIO'].str.strip().str.lower().dropna().unique())
barrios_raw  = set(raw['BARRIO'].str.strip().str.lower().dropna().unique())

no_en_clima = barrios_acc - barrios_cli
print(f"\\nBarrios en accidentes sin datos de clima: {len(no_en_clima)}")
print(f"  {sorted(no_en_clima)}")
print(f"\\n→ Estos {len(no_en_clima)} barrios pierden {(df['BARRIO'].isin(no_en_clima) & (df['target']==1)).sum()} positivos (se excluyen al hacer LEFT JOIN desde clima).")
"""))

cells.append(code("""# ── Tratamiento de nulos — Decisiones documentadas ──────────

# 1. windBearing (12.9%): sin dirección de viento → indicativo de calma → imputar con mediana
df['windBearing'] = df['windBearing'].fillna(df.groupby('BARRIO')['windBearing'].transform('median'))
df['windBearing'] = df['windBearing'].fillna(df['windBearing'].median())

# 2. summary/icon (6.9%): categóricas → asignar "Unknown"
df['summary'] = df['summary'].fillna('Unknown')
df['icon']    = df['icon'].fillna('unknown')

# 3. precipIntensity / precipProbability (6.9%): sin lluvia registrada → 0
df['precipIntensity']   = df['precipIntensity'].fillna(0)
df['precipProbability'] = df['precipProbability'].fillna(0)

# 4. Demás numéricas (<0.5%): imputar con mediana global
cols_num_clima = ['temperature','apparentTemperature','dewPoint','humidity',
                  'windSpeed','cloudCover','uvIndex','visibility']
for col in cols_num_clima:
    df[col] = df[col].fillna(df[col].median())

nulos_post = df.isnull().sum().sum()
print(f"✓ Tratamiento de nulos completado")
print(f"  Nulos restantes: {nulos_post}")
print()
print("Decisiones tomadas:")
print("  • windBearing     → mediana por barrio (sin brisa = sin dirección)")
print("  • summary / icon  → categoría 'Unknown'")
print("  • precipIntensity → 0 (sin precipitación registrada)")
print("  • temp, humidity… → mediana global (< 0.5% nulos)")
"""))

cells.append(code("""# ── Outliers: boxplots de variables climáticas clave ──────────
vars_box = ['temperature','humidity','precipIntensity','windSpeed','visibility']

fig, axes = plt.subplots(1, len(vars_box), figsize=(18, 5))
fig.suptitle('4.2 — Outliers en Variables Climáticas', fontweight='bold')

sample = df.sample(n=min(100000, len(df)), random_state=42)
for i, var in enumerate(vars_box):
    axes[i].boxplot(sample[var].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='#3498db', color='white'),
                    medianprops=dict(color='#e74c3c', linewidth=2))
    axes[i].set_title(var); axes[i].set_xticks([])

plt.tight_layout(); plt.show()
print("🔍 precipIntensity y windSpeed tienen valores extremos esperables (tormentas, vendavales).")
print("   Se conservan como información real del clima.")
"""))

# Guardar
with open('nb_parts/cells_p2.pkl', 'wb') as f:
    pickle.dump(cells, f)

print(f"✓ Parte 2 lista: {len(cells)} celdas totales")
