"""
Modelo Preditivo de Ocorrências Criminais - SSP-SP
Regressão para prever total de ocorrências por delegacia/região/mês
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── 1. DADOS ──────────────────────────────────────────────────────────────────

print("=" * 60)
print("MODELO PREDITIVO DE OCORRÊNCIAS CRIMINAIS - SSP-SP")
print("=" * 60)

if not os.path.exists("data/crimes_ssp_sp.csv"):
    print("\nGerando dataset...")
    exec(open("data/gerar_dados_ssp.py").read())

df = pd.read_csv("data/crimes_ssp_sp.csv")
print(f"\nDataset carregado: {df.shape[0]} registros, {df.shape[1]} colunas")
print(df.head())

# ── 2. ANÁLISE EXPLORATÓRIA ───────────────────────────────────────────────────

print("\n--- Estatísticas da variável-alvo ---")
print(df["total_ocorrencias"].describe().round(2))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Análise Exploratória - Ocorrências Criminais SP", fontsize=14, fontweight="bold")

# Distribuição da variável-alvo
axes[0, 0].hist(df["total_ocorrencias"], bins=40, edgecolor="white", color="#2196F3")
axes[0, 0].set_title("Distribuição das Ocorrências")
axes[0, 0].set_xlabel("Total de Ocorrências")
axes[0, 0].set_ylabel("Frequência")

# Média por mês (sazonalidade)
media_mes = df.groupby("mes")["total_ocorrencias"].mean()
axes[0, 1].plot(media_mes.index, media_mes.values, marker="o", color="#4CAF50", linewidth=2)
axes[0, 1].set_title("Sazonalidade Mensal (média)")
axes[0, 1].set_xlabel("Mês")
axes[0, 1].set_ylabel("Média de Ocorrências")
axes[0, 1].set_xticks(range(1, 13))

# Top 10 delegacias
top10 = df.groupby("delegacia")["total_ocorrencias"].mean().nlargest(10)
axes[1, 0].barh(top10.index, top10.values, color="#FF5722")
axes[1, 0].set_title("Top 10 Delegacias (média mensal)")
axes[1, 0].set_xlabel("Média de Ocorrências")

# Correlação com variáveis numéricas
num_cols = ["populacao", "densidade_demografica", "idh", "total_ocorrencias"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1], vmin=-1, vmax=1)
axes[1, 1].set_title("Correlação entre Features")

plt.tight_layout()
plt.savefig("outputs/01_analise_exploratoria.png", dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico salvo: outputs/01_analise_exploratoria.png")

# ── 3. PREPARAÇÃO DOS DADOS ───────────────────────────────────────────────────

le = LabelEncoder()
df["delegacia_enc"] = le.fit_transform(df["delegacia"])

FEATURES = [
    "ano", "mes", "mes_sin", "mes_cos", "trimestre",
    "delegacia_enc", "populacao", "densidade_demografica", "idh",
    "furtos", "roubos", "lesao_corporal", "estelionato", "trafico_drogas",
]
TARGET = "total_ocorrencias"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTreino: {len(X_train)} | Teste: {len(X_test)}")

# ── 4. MODELOS ────────────────────────────────────────────────────────────────

modelos = {
    "Regressão Linear": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
}

resultados = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Treinando modelos ---")
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(modelo, X_train, y_train, cv=kf, scoring="r2").mean()

    resultados[nome] = {"MAE": mae, "RMSE": rmse, "R²": r2, "CV R²": cv, "modelo": modelo, "pred": y_pred}
    print(f"  {nome:25s} | MAE={mae:7.1f} | RMSE={rmse:7.1f} | R²={r2:.4f} | CV R²={cv:.4f}")

# ── 5. MELHOR MODELO ──────────────────────────────────────────────────────────

melhor_nome = max(resultados, key=lambda k: resultados[k]["R²"])
melhor = resultados[melhor_nome]
print(f"\nMelhor modelo: {melhor_nome} (R² = {melhor['R²']:.4f})")

# ── 6. VISUALIZAÇÕES DO MELHOR MODELO ─────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"Avaliação do Melhor Modelo: {melhor_nome}", fontsize=13, fontweight="bold")

# Real vs Previsto
axes[0].scatter(y_test, melhor["pred"], alpha=0.4, color="#2196F3", s=15)
lim = [min(y_test.min(), melhor["pred"].min()), max(y_test.max(), melhor["pred"].max())]
axes[0].plot(lim, lim, "r--", linewidth=1.5)
axes[0].set_xlabel("Real")
axes[0].set_ylabel("Previsto")
axes[0].set_title(f"Real vs Previsto\nR² = {melhor['R²']:.4f}")

# Resíduos
residuos = y_test.values - melhor["pred"]
axes[1].scatter(melhor["pred"], residuos, alpha=0.4, color="#FF5722", s=15)
axes[1].axhline(0, color="black", linewidth=1.5, linestyle="--")
axes[1].set_xlabel("Valor Previsto")
axes[1].set_ylabel("Resíduo")
axes[1].set_title("Resíduos")

# Comparação de modelos (R²)
nomes = list(resultados.keys())
r2s   = [resultados[n]["R²"] for n in nomes]
cores = ["#4CAF50" if n == melhor_nome else "#90CAF9" for n in nomes]
bars  = axes[2].barh(nomes, r2s, color=cores)
axes[2].set_xlabel("R²")
axes[2].set_title("Comparação de Modelos (R²)")
axes[2].set_xlim(0, 1.05)
for bar, val in zip(bars, r2s):
    axes[2].text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/02_avaliacao_modelo.png", dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico salvo: outputs/02_avaliacao_modelo.png")

# ── 7. IMPORTÂNCIA DAS FEATURES (Random Forest) ───────────────────────────────

rf_nome = "Random Forest"
rf_modelo = resultados[rf_nome]["modelo"]
importancias = rf_modelo.feature_importances_
feat_imp = pd.Series(importancias, index=FEATURES).sort_values(ascending=True)

plt.figure(figsize=(9, 6))
feat_imp.plot(kind="barh", color="#2196F3", edgecolor="white")
plt.title("Importância das Features (Random Forest)", fontsize=13, fontweight="bold")
plt.xlabel("Importância")
plt.tight_layout()
plt.savefig("outputs/03_importancia_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico salvo: outputs/03_importancia_features.png")

# ── 8. PREVISÃO FUTURA ────────────────────────────────────────────────────────

print("\n--- Previsão para 2024 (próximos 12 meses) ---")
modelo_final = melhor["modelo"]

# Usa médias históricas por delegacia como base para os crimes sub-tipos
medias_hist = df.groupby("delegacia")[["furtos", "roubos", "lesao_corporal", "estelionato", "trafico_drogas"]].mean().round(0).astype(int)

registros_fut = []
for _, row in df[["delegacia", "delegacia_enc", "populacao", "densidade_demografica", "idh"]].drop_duplicates().iterrows():
    del_name = row["delegacia"]
    medias   = medias_hist.loc[del_name]
    for mes in range(1, 13):
        registros_fut.append({
            "ano": 2024,
            "mes": mes,
            "mes_sin": np.sin(2 * np.pi * mes / 12),
            "mes_cos": np.cos(2 * np.pi * mes / 12),
            "trimestre": ((mes - 1) // 3) + 1,
            "delegacia_enc": row["delegacia_enc"],
            "populacao": row["populacao"],
            "densidade_demografica": row["densidade_demografica"],
            "idh": row["idh"],
            "furtos": medias["furtos"],
            "roubos": medias["roubos"],
            "lesao_corporal": medias["lesao_corporal"],
            "estelionato": medias["estelionato"],
            "trafico_drogas": medias["trafico_drogas"],
            "delegacia": del_name,
        })

df_futuro = pd.DataFrame(registros_fut)
df_futuro["previsao_ocorrencias"] = modelo_final.predict(df_futuro[FEATURES]).round(0).astype(int)

resumo_futuro = df_futuro.groupby("delegacia")["previsao_ocorrencias"].sum().sort_values(ascending=False)
print(resumo_futuro.to_string())

df_futuro.to_csv("outputs/previsoes_2024.csv", index=False)
print("\nPrevisões salvas: outputs/previsoes_2024.csv")

# Plot previsão
fig, ax = plt.subplots(figsize=(10, 6))
top5 = resumo_futuro.head(5).index.tolist()
for del_name in top5:
    subset = df_futuro[df_futuro["delegacia"] == del_name].sort_values("mes")
    ax.plot(subset["mes"], subset["previsao_ocorrencias"], marker="o", label=del_name)

ax.set_title("Previsão Mensal de Ocorrências - 2024 (Top 5 Delegacias)", fontsize=12, fontweight="bold")
ax.set_xlabel("Mês")
ax.set_ylabel("Ocorrências Previstas")
ax.set_xticks(range(1, 13))
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/04_previsao_2024.png", dpi=150, bbox_inches="tight")
plt.close()
print("Gráfico salvo: outputs/04_previsao_2024.png")

# ── 9. EXPORTAR MODELO ────────────────────────────────────────────────────────

joblib.dump(modelo_final, "models/modelo_ocorrencias_sp.pkl")
joblib.dump(le, "models/label_encoder_delegacias.pkl")
print(f"\nModelo exportado: models/modelo_ocorrencias_sp.pkl")

# ── 10. RELATÓRIO FINAL ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RELATÓRIO FINAL")
print("=" * 60)
print(f"Melhor modelo    : {melhor_nome}")
print(f"R² (teste)       : {melhor['R²']:.4f}")
print(f"MAE              : {melhor['MAE']:.1f} ocorrências")
print(f"RMSE             : {melhor['RMSE']:.1f} ocorrências")
print(f"CV R² (5-fold)   : {melhor['CV R²']:.4f}")
print(f"\nTotal previsto 2024: {df_futuro['previsao_ocorrencias'].sum():,} ocorrências")
print("\nArquivos gerados:")
print("  outputs/01_analise_exploratoria.png")
print("  outputs/02_avaliacao_modelo.png")
print("  outputs/03_importancia_features.png")
print("  outputs/04_previsao_2024.png")
print("  outputs/previsoes_2024.csv")
print("  models/modelo_ocorrencias_sp.pkl")
print("=" * 60)
