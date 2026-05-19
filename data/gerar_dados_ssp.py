"""
Gera dataset sintético baseado na estrutura dos dados públicos da SSP-SP.
Referência real: http://www.ssp.sp.gov.br/transparenciassp/
"""

import numpy as np
import pandas as pd

np.random.seed(42)

DELEGACIAS = {
    "Centro":         {"populacao": 371_000, "densidade": 14_200, "idh": 0.84, "base_crimes": 4200},
    "Sé":             {"populacao": 23_000,  "densidade": 15_000, "idh": 0.82, "base_crimes": 1800},
    "Brás":           {"populacao": 31_000,  "densidade": 12_000, "idh": 0.78, "base_crimes": 1400},
    "Lapa":           {"populacao": 65_000,  "densidade": 7_800,  "idh": 0.86, "base_crimes": 1100},
    "Pinheiros":      {"populacao": 72_000,  "densidade": 9_100,  "idh": 0.91, "base_crimes": 900},
    "Vila Mariana":   {"populacao": 141_000, "densidade": 11_200, "idh": 0.89, "base_crimes": 1300},
    "Ipiranga":       {"populacao": 108_000, "densidade": 8_500,  "idh": 0.82, "base_crimes": 1200},
    "Santo André":    {"populacao": 720_000, "densidade": 4_200,  "idh": 0.80, "base_crimes": 5800},
    "Guarulhos":      {"populacao": 1_400_000,"densidade": 3_900, "idh": 0.76, "base_crimes": 9200},
    "Osasco":         {"populacao": 700_000, "densidade": 10_200, "idh": 0.76, "base_crimes": 6100},
    "São Bernardo":   {"populacao": 840_000, "densidade": 2_900,  "idh": 0.80, "base_crimes": 6400},
    "Mauá":           {"populacao": 480_000, "densidade": 3_600,  "idh": 0.74, "base_crimes": 4700},
    "Diadema":        {"populacao": 420_000, "densidade": 13_000, "idh": 0.75, "base_crimes": 4300},
    "Carapicuíba":    {"populacao": 390_000, "densidade": 11_500, "idh": 0.73, "base_crimes": 4100},
    "Taboão da Serra":{"populacao": 290_000, "densidade": 10_800, "idh": 0.73, "base_crimes": 3200},
    "Itaquera":       {"populacao": 200_000, "densidade": 6_400,  "idh": 0.77, "base_crimes": 2800},
    "Penha":          {"populacao": 130_000, "densidade": 8_900,  "idh": 0.79, "base_crimes": 1900},
    "Mooca":          {"populacao": 115_000, "densidade": 8_100,  "idh": 0.83, "base_crimes": 1500},
    "Santana":        {"populacao": 97_000,  "densidade": 7_300,  "idh": 0.85, "base_crimes": 1200},
    "Campo Limpo":    {"populacao": 160_000, "densidade": 9_600,  "idh": 0.74, "base_crimes": 2400},
}

ANOS = range(2018, 2024)
MESES = range(1, 13)

registros = []

for delegacia, info in DELEGACIAS.items():
    for ano in ANOS:
        for mes in MESES:
            # Sazonalidade: mais crimes no verão/fim de ano (dez-jan) e menos em jun-jul
            sazonalidade = 1.0 + 0.15 * np.sin(2 * np.pi * (mes - 3) / 12)

            # Tendência temporal (leve redução ao longo dos anos)
            tendencia = 1.0 - 0.02 * (ano - 2018)

            # IDH inversamente proporcional à criminalidade
            fator_idh = 2.0 - info["idh"]

            # Densidade urbana aumenta crimes
            fator_densidade = 1.0 + (info["densidade"] - 5000) / 50000

            # Cálculo base com ruído
            media = (info["base_crimes"] / 12) * sazonalidade * tendencia * fator_idh * fator_densidade
            ocorrencias = int(np.random.normal(media, media * 0.08))
            ocorrencias = max(0, ocorrencias)

            # Distribuição por tipo de crime (soma = ocorrencias)
            pesos = np.array([0.30, 0.22, 0.18, 0.12, 0.10, 0.08])
            tipos = np.random.multinomial(ocorrencias, pesos)

            registros.append({
                "ano": ano,
                "mes": mes,
                "delegacia": delegacia,
                "populacao": info["populacao"],
                "densidade_demografica": info["densidade"],
                "idh": info["idh"],
                "furtos": tipos[0],
                "roubos": tipos[1],
                "homicidios_dolosos": max(0, int(tipos[2] * 0.04)),
                "lesao_corporal": tipos[3],
                "estelionato": tipos[4],
                "trafico_drogas": tipos[5],
                "total_ocorrencias": ocorrencias,
            })

df = pd.DataFrame(registros)

# Features derivadas
df["trimestre"] = ((df["mes"] - 1) // 3) + 1
df["taxa_por_100k"] = (df["total_ocorrencias"] / df["populacao"]) * 100_000
df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

df.to_csv("data/crimes_ssp_sp.csv", index=False)
print(f"Dataset gerado: {len(df)} registros")
print(df.head())
print("\nEstatísticas básicas:")
print(df["total_ocorrencias"].describe())
