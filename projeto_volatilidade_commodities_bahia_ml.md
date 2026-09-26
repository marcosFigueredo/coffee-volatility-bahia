# Projeto de Estudo

## Previsão de Volatilidade do Café (Arábica e Conilon) na Bahia com Machine Learning

### Título provisório em inglês
**Cross-Market Machine Learning for Coffee Price Volatility Forecasting: Evidence from Arabica and Conilon in Bahia, Brazil**

---

## 1. Introduction

Volatility forecasting is a central task in financial risk management, particularly in commodity markets. Reliable forecasts support hedging, margining, position sizing, contract negotiation, and the monitoring of revenue and market risk (Poon and Granger, 2003; Andersen et al., 2003). Coffee markets are especially challenging in this regard because price uncertainty reflects the interaction of weather shocks, production cycles, inventories, exchange rates, energy costs, international futures markets, and changes in demand. These drivers operate across different time scales and may generate nonlinear, asymmetric, and regime-dependent volatility dynamics.

Coffee provides a relevant empirical setting for investigating these issues because Arabica and Conilon/Robusta are economically related but structurally distinct products. Arabica is closely associated with the ICE Coffee C benchmark and with supply conditions in Brazil and other Arabica-producing regions, whereas Conilon is connected to the Robusta market and to industrial demand. In Bahia, these two varieties are traded in local markets with different regional characteristics. The SEAGRI-BA series for Vitória da Conquista and Luís Eduardo Magalhães represent Arabica markets, while the Eunápolis series represents Conilon. Local prices may contain regional information that is not fully reflected in international benchmarks, while remaining exposed to common financial and commodity-market shocks. This combination makes Bahia a useful laboratory for studying the transmission of multisource market information into local commodity volatility.

Econometric models such as GARCH and EGARCH remain important benchmarks because they represent volatility clustering and, in some specifications, asymmetric responses to shocks (Engle, 1982; Bollerslev, 1986; Nelson, 1991). HAR-type models are also useful for representing persistence across volatility horizons when their data requirements are satisfied (Corsi, 2009). However, these models impose relatively restrictive functional forms and may not fully capture interactions among heterogeneous markets, nonlinear effects, or changing predictive relationships. Machine-learning models offer a flexible alternative for learning such patterns, but their flexibility does not guarantee superior forecasting performance. Financial time series are nonstationary, heavy-tailed, and vulnerable to structural breaks, while limited regional samples increase the risk of overfitting. Consequently, claims about the value of machine learning require strictly chronological out-of-sample evaluation and explicit control of data leakage.

Although machine learning has increasingly been applied to financial and agricultural forecasting, the existing literature provides limited evidence on the incremental predictive value of cross-market and financial information for the volatility of local coffee prices. Recent coffee studies have largely focused on price-level forecasting or on global and monthly datasets, while related volatility studies have emphasized parametric risk models, hybrid architectures, and comparisons among algorithms (Deina et al., 2022; Brun, 2026; Manogna et al., 2025; Byrareddy et al., 2026). Less attention has been given to the joint comparison of Arabica and Conilon/Robusta local markets, the separation of own-market, cross-commodity, and financial information blocks, and the evaluation of their contribution across forecast horizons and volatility regimes. The research gap is therefore not simply the application of machine learning to coffee, but the systematic assessment of whether heterogeneous financial-market information improves the prediction of local coffee volatility under a realistic forecasting design.

Against this background, this study investigates whether cross-market and financial information improves the out-of-sample forecasting of coffee price volatility in local Arabica and Conilon markets in Bahia, Brazil. We integrate local coffee prices with national and international coffee references, coffee futures, agricultural commodity futures, exchange rates, market-uncertainty indicators, energy prices, and interest-rate variables. We compare econometric benchmarks with representative machine-learning models using four information sets: own-market history, own-market history plus cross-commodity variables, own-market history plus financial variables, and the complete predictor set. Forecasts are evaluated at multiple horizons using a strictly chronological expanding-window design, with QLIKE as the primary volatility-loss function and MAE, RMSE, and Diebold–Mariano tests as complementary assessments. Explainability analyses are used to identify predictive drivers; their results are interpreted as evidence of predictive contribution, not as causal effects.

The study makes three main contributions. First, it develops an integrated multisource and multimarket machine-learning framework for forecasting volatility in local coffee markets, linking regional price information to international commodity and financial signals. Second, it quantifies the incremental predictive value of cross-commodity and financial information through controlled ablation experiments rather than relying only on a ranking of algorithms. Third, it examines whether predictive gains and relevant variables differ between Arabica and Conilon markets, across forecast horizons, and across low-, normal-, and high-volatility regimes. The empirical setting is regional, but the forecasting design is intended to provide transferable evidence for data-driven commodity-risk monitoring in emerging markets.

The remainder of the article is organized as follows. Section 2 reviews the related literature. Section 3 describes the data and the construction of the volatility measures. Section 4 presents the methodology, models, and temporal validation strategy. Section 5 describes the experimental design. Section 6 reports the results. Section 7 presents the explainability analysis, and Section 8 discusses the implications and limitations of the findings. Section 9 concludes the article.

> **Project note (not for the manuscript; updated on 2026-09-09):** The original project covered soybean, cotton, cocoa, and coffee as target commodities. The data audit (`data_audit.csv`) showed that only the local coffee series met the minimum continuity criteria for target modeling: Conilon/Eunápolis is comparatively complete, while Arabica/Vitória da Conquista and Arabica/Luís Eduardo Magalhães present moderate gaps of approximately 33–37%. Local soybean, cotton, and cocoa series contain gaps ranging from 400 to more than 4000 days and were therefore excluded as independent targets. They remain in the study only as cross-commodity explanatory variables through usable international futures series.

---

## 2. Problema de pesquisa

> **As informações cross-commodity (outras culturas agrícolas) e dos mercados financeiros melhoram significativamente a previsão da volatilidade futura do café Arábica e Conilon nas praças baianas?**

---

## 3. Perguntas de pesquisa

- **RQ1:** Modelos de Machine Learning apresentam desempenho superior aos modelos econométricos tradicionais na previsão da volatilidade do café?
- **RQ2:** A inclusão de informações provenientes de outras commodities (soja, algodão, cacau) melhora a previsão?
- **RQ3:** Variáveis financeiras, como USD/BRL, VIX e Brent, fornecem ganho preditivo adicional?
- **RQ4:** Os ganhos dos modelos de ML são diferentes em regimes de baixa, média e alta volatilidade?
- **RQ5:** Quais variáveis apresentam maior contribuição para a previsão de cada variedade de café?
- **RQ6:** Os fatores relevantes são semelhantes entre café Arábica e Conilon ou específicos de cada variedade/praça?

---

## 4. Hipóteses

- **H1:** Modelos de Machine Learning apresentam menor erro de previsão da volatilidade do que modelos econométricos tradicionais.
- **H2:** A incorporação de informações cross-commodity (soja, algodão, cacau) melhora significativamente a previsão.
- **H3:** A incorporação de informações financeiras e macrofinanceiras melhora adicionalmente o desempenho.
- **H4:** Os ganhos das variáveis cross-market são maiores durante períodos de elevada volatilidade.
- **H5:** A importância das variáveis preditoras varia entre Arábica e Conilon e entre regimes.

---

## 5. Objetivo geral

Desenvolver e avaliar modelos de Machine Learning para previsão da volatilidade futura do café (Arábica e Conilon) nas praças baianas, investigando o valor incremental de informações provenientes de outras commodities agrícolas e dos mercados financeiros.

---

## 6. Objetivos específicos

1. Construir uma base temporal integrada com preços locais (café), nacionais e internacionais de commodities e variáveis financeiras.
2. Avaliar qualidade, frequência e continuidade das séries.
3. Construir medidas de retorno e volatilidade em múltiplos horizontes.
4. Comparar modelos econométricos e modelos de Machine Learning.
5. Quantificar o ganho proporcionado por informações cross-commodity (soja, algodão, cacau).
6. Quantificar o ganho proporcionado por informações financeiras.
7. Avaliar os modelos em diferentes regimes de volatilidade.
8. Identificar as variáveis mais relevantes por métodos de explicabilidade.
9. Testar estatisticamente as diferenças de desempenho entre modelos.
10. Comparar a dinâmica de volatilidade entre Arábica e Conilon.
11. Produzir um framework reprodutível de previsão de risco para o café baiano.

---

## 7. Commodities analisadas

**Commodities-alvo (variável dependente):**
- café Arábica;
- café Conilon/Robusta.

**Commodities cross-market (variáveis explicativas apenas, não são alvo):**
- soja;
- algodão;
- cacau.

Essa divisão foi definida a partir da auditoria de dados (`data_audit.csv`, seção 33): as praças-alvo de soja/algodão/cacau no SEAGRI-BA têm gaps de 400 a mais de 4000 dias e foram descartadas como alvo; café Arábica e Conilon foram as únicas com continuidade suficiente para modelagem.

---

## 8. Fontes de dados

### 8.1 SEAGRI-BA
Séries locais (alvo):
- café Arábica — Vitória da Conquista;
- café Arábica — Luís Eduardo Magalhães;
- café Conilon — Eunápolis.

### 8.2 CEPEA
Séries brasileiras de referência para café Arábica e café Robusta (nacional, benchmark/fallback — download automático ainda bloqueado por proteção anti-bot do site; ver seção 33).

### 8.3 Mercados futuros
Referências internacionais (café = alvo internacional/benchmark; soja, cacau e algodão = variáveis cross-market):
- café — ICE Coffee C;
- soja — CME Soybean Futures (proxy: Yahoo Finance `ZS=F`);
- cacau — ICE Cocoa Futures (proxy: Yahoo Finance `CC=F`);
- algodão — ICE Cotton No. 2 (proxy: Yahoo Finance `CT=F`).

### 8.4 Banco Central do Brasil
- USD/BRL;
- SELIC.

### 8.5 FRED / CBOE / EIA
Variáveis candidatas:
- VIX;
- Brent;
- Dollar Index;
- Federal Funds Rate;
- Treasury 10 anos.

### 8.6 Yahoo Finance
Fonte complementar:
- `ZS=F` — soja;
- `CC=F` — cacau;
- `KC=F` — café;
- `CT=F` — algodão;
- `BZ=F` — Brent;
- `CL=F` — WTI;
- `^VIX` — VIX;
- `BRL=X` — USD/BRL;
- `^GSPC` — S&P 500;
- `^BVSP` — Ibovespa.

---

## 9. Etapa 1 — Auditoria dos dados

Antes da modelagem, cada CSV deverá ser auditado quanto a:

- fonte;
- commodity;
- praça;
- unidade;
- primeira e última data;
- número de observações;
- frequência;
- percentual de ausentes;
- duplicatas;
- percentual de preços repetidos;
- maior intervalo sem observação;
- número médio de observações por mês;
- mudanças de unidade;
- valores anômalos;
- percentual de dias úteis disponíveis.

### Produto
Gerar `data_audit.csv` com estrutura semelhante a:

```text
source
series
commodity
location
start_date
end_date
n_obs
missing_pct
duplicate_pct
unchanged_price_pct
max_gap_days
frequency
usable
notes
```

---

## 10. Critério para seleção das séries

Uma série será adequada para modelagem diária se apresentar:

- horizonte temporal suficientemente longo;
- frequência compatível com dias úteis;
- baixa proporção de ausentes;
- poucas lacunas longas;
- variação de preço suficiente para produzir volatilidade informativa.

Se a série local SEAGRI-BA tiver baixa densidade:
- **Alternativa A:** trabalhar em frequência semanal;
- **Alternativa B:** usar CEPEA/internacional como alvo principal e SEAGRI-BA para validação regional ou análise de transmissão.

---

## 11. Construção dos retornos

Para preço \(P_t\):

\[
r_t = 100 \ln\left(\frac{P_t}{P_{t-1}}\right)
\]

Os modelos trabalharão preferencialmente com retornos e volatilidade, não com preços brutos.

---

## 12. Variável-alvo: volatilidade futura

Para horizonte \(h\):

\[
RV_{t,h}=\sqrt{\sum_{i=1}^{h}r_{t+i}^{2}}
\]

Horizontes iniciais:

\[
h=1,5,20
\]

Prioridade analítica: 5 e 20 dias.

---

## 13. Features

### 13.1 Histórico próprio
- retornos defasados;
- volatilidade de 5, 10 e 20 dias;
- médias móveis;
- desvio-padrão móvel;
- amplitude high-low;
- volume;
- open interest;
- momentum.

### 13.2 Cross-commodity
Para cada commodity-alvo:
- retornos das demais commodities;
- volatilidade das demais commodities.

### 13.3 Financeiras
- USD/BRL;
- retorno e volatilidade do USD/BRL;
- VIX;
- variação do VIX;
- Brent;
- WTI;
- Dollar Index;
- S&P 500;
- Ibovespa;
- juros brasileiros;
- juros norte-americanos.

---

## 14. Experimentos

### E0 — Baseline econométrico
- GARCH(1,1);
- EGARCH;
- HAR-RV, quando aplicável.

### E1 — ML univariado
\[
X_t = \text{histórico da própria commodity}
\]

### E2 — ML cross-commodity
\[
X_t = \{\text{commodity-alvo} + \text{demais commodities}\}
\]

### E3 — ML financeiro
\[
X_t = \{\text{commodity-alvo} + \text{variáveis financeiras}\}
\]

### E4 — Modelo completo
\[
X_t = \{\text{commodity-alvo} + \text{cross-commodity} + \text{financeiro}\}
\]

---

## 15. Modelos

### Econométricos
- GARCH;
- EGARCH;
- HAR-RV.

### Machine Learning
- LightGBM;
- Random Forest;
- XGBoost.

### Deep Learning
- LSTM;
- TCN.

Transformer poderá ser incorporado apenas se o volume de dados e os resultados justificarem.

---

## 16. Regimes de mercado

Inicialmente:

\[
R_t \in \{Low, Normal, High\}
\]

Definição por quantis da volatilidade:
- Low: abaixo do percentil 33;
- Normal: percentis 33–66;
- High: acima do percentil 66.

Como robustez, poderá ser usado Hidden Markov Model.

---

## 17. Estratégia de validação

Não será usado split aleatório.

### Expanding window

```text
Treino 2010–2018 -> Teste 2019
Treino 2010–2019 -> Teste 2020
Treino 2010–2020 -> Teste 2021
Treino 2010–2021 -> Teste 2022
...
```

Também poderá ser avaliado rolling window.

---

## 18. Prevenção de data leakage

Todas as transformações devem respeitar a ordem temporal:

- normalização;
- seleção de variáveis;
- imputação;
- cálculo de features;
- tuning;
- identificação de regimes.

Nenhuma informação do teste pode entrar no treinamento ou preparação das features.

---

## 19. Métricas

Principais:

### QLIKE
Métrica central para avaliação de previsão de volatilidade.

### MAE

\[
MAE = \frac{1}{n}\sum |y_t-\hat{y}_t|
\]

### RMSE

\[
RMSE = \sqrt{\frac{1}{n}\sum(y_t-\hat{y}_t)^2}
\]

---

## 20. Comparação estatística

Usar teste de Diebold-Mariano para:

- GARCH × LightGBM;
- HAR-RV × LightGBM;
- E1 × E2;
- E1 × E3;
- E1 × E4;
- E2 × E4.

---

## 21. Explicabilidade

Aplicar SHAP ao melhor modelo.

Perguntas:
- USD/BRL é determinante para todas as commodities?
- VIX ganha importância em períodos turbulentos?
- Brent afeta mais soja e algodão?
- outras commodities antecipam a volatilidade da commodity-alvo?
- a importância muda entre regimes?

---

## 22. Análises de ablação

```text
M0 = own history
M1 = own history + cross-commodity
M2 = own history + financial
M3 = own history + cross-commodity + financial
```

Ganho cross-commodity:

\[
\Delta_{cross}=Error(M0)-Error(M1)
\]

Ganho financeiro:

\[
\Delta_{fin}=Error(M0)-Error(M2)
\]

Ganho total:

\[
\Delta_{all}=Error(M0)-Error(M3)
\]

---

## 23. Robustez

1. diferentes horizontes;
2. diferentes períodos de treinamento;
3. expanding versus rolling window;
4. pré e pós-pandemia;
5. baixa versus alta volatilidade;
6. exclusão de períodos extremos;
7. diferentes medidas de volatilidade;
8. somente fontes oficiais;
9. inclusão de fontes complementares;
10. comparação local, nacional e internacional.

---

## 24. Possível extensão multiescala

Se o estudo principal for promissor, pode-se incorporar wavelets:

\[
X_t=X_t^{low}+X_t^{medium}+X_t^{high}
\]

A hipótese seria que diferentes escalas temporais possuem valor preditivo distinto.

Essa extensão deve permanecer secundária para não aumentar desnecessariamente a complexidade inicial.

---

## 25. Estrutura do dataset integrado

```text
date
target_variety          # arabica | conilon
target_location
price
return_1d
rv_5
rv_10
rv_20
cepea_price
ice_coffee_c_close       # futuro internacional (benchmark do café)
ice_coffee_c_return
usd_brl
usd_brl_return
usd_brl_vol
vix
vix_change
brent
brent_return
other_variety_return     # conilon quando alvo=arabica, e vice-versa
other_variety_vol_5
soy_return               # cross-commodity (não-alvo)
cotton_return
cocoa_return
soy_vol_5
cotton_vol_5
cocoa_vol_5
```

---

## 26. Pipeline

```text
CSV brutos
    |
    v
Auditoria
    |
    v
Padronização
    |
    v
Sincronização temporal
    |
    v
Retornos e volatilidade
    |
    +--------------------+
    |                    |
    v                    v
Cross-commodity      Financeiro
    |                    |
    +---------+----------+
              |
              v
       Dataset integrado
              |
              v
      Walk-forward split
              |
     +--------+---------+
     |        |         |
   GARCH   LightGBM    TCN/LSTM
     |        |         |
     +--------+---------+
              |
              v
       QLIKE / MAE / RMSE
              |
              v
       Diebold-Mariano
              |
              v
             SHAP
              |
              v
    Análise por regime
```

---

## 27. Contribuição científica esperada

A contribuição central será:

> **Quantificar quanto da previsibilidade da volatilidade de commodities agrícolas é proveniente do próprio histórico, de informações cross-commodity e de variáveis financeiras, e verificar como essa contribuição muda em diferentes regimes de mercado.**

\[
\text{Own information}
\rightarrow
\text{Cross-commodity}
\rightarrow
\text{Financial information}
\rightarrow
\text{Volatility forecast}
\]

---

## 28. Contribuição regional

A Bahia funcionará como laboratório empírico de uma grande economia agrícola brasileira.

Possíveis resultados:
- indicadores de risco para commodities estratégicas;
- evidências sobre integração entre mercados locais e internacionais;
- identificação de exposição cambial;
- identificação de períodos de maior risco;
- informação para produtores, cooperativas e agentes de comercialização;
- base para futuras ferramentas de apoio à decisão.

---

## 29. Estrutura prevista do artigo

1. Introduction
2. Related Work
3. Data
4. Methodology
5. Experiments
6. Results
7. Explainability
8. Discussion
9. Conclusion

---

## 30. Tabelas esperadas

1. descrição das séries e fontes;
2. estatísticas descritivas;
3. qualidade dos dados;
4. resultados globais;
5. resultados por commodity;
6. resultados por horizonte;
7. ablation study;
8. resultados por regime;
9. testes Diebold-Mariano.

---

## 31. Figuras esperadas

1. pipeline metodológico;
2. evolução histórica dos preços;
3. retornos;
4. volatilidade realizada;
5. correlação entre mercados;
6. desempenho por horizonte;
7. desempenho por regime;
8. SHAP summary plot;
9. importância das variáveis por commodity.

---

## 32. Ordem prática de execução

### Fase 1 — Dados
1. verificar os CSVs;
2. gerar auditoria;
3. selecionar séries;
4. harmonizar datas e unidades;
5. gerar dataset mestre.

### Fase 2 — Exploração
6. calcular retornos;
7. calcular volatilidades;
8. verificar distribuições;
9. avaliar correlações;
10. identificar choques.

### Fase 3 — Baselines
11. naive volatility;
12. GARCH;
13. EGARCH;
14. HAR-RV.

### Fase 4 — ML
15. Random Forest;
16. XGBoost;
17. LightGBM;
18. LSTM;
19. TCN.

### Fase 5 — Experimentos
20. E0;
21. E1;
22. E2;
23. E3;
24. E4.

### Fase 6 — Avaliação
25. QLIKE;
26. MAE;
27. RMSE;
28. Diebold-Mariano;
29. análise por regime.

### Fase 7 — Interpretação
30. SHAP;
31. importância por commodity;
32. importância por regime;
33. discussão econômica.

---

## 33. Primeira tarefa computacional

A próxima etapa deve ser criar:

`01_audit_data.py`

Responsabilidades:

1. percorrer todos os CSVs;
2. detectar colunas de data;
3. identificar período disponível;
4. contar observações;
5. detectar duplicatas;
6. identificar gaps;
7. estimar frequência;
8. calcular percentual de ausentes;
9. medir repetição de preços;
10. produzir `data_audit.csv`.

Somente após essa auditoria deve ser definida a lista definitiva de commodities e fontes.

---

## 34. Critério de sucesso

O estudo será particularmente relevante se demonstrar pelo menos um dos seguintes resultados:

1. ML supera consistentemente GARCH/HAR em previsão out-of-sample;
2. informação cross-commodity melhora significativamente a previsão;
3. variáveis financeiras produzem ganho significativo em períodos turbulentos;
4. os determinantes da volatilidade diferem entre commodities;
5. modelos integrados antecipam melhor episódios de alta volatilidade;
6. séries locais baianas apresentam comportamento distinto dos benchmarks nacionais ou internacionais.

---

## 35. Resultado final esperado

O principal resultado não deverá ser apenas um ranking de algoritmos.

O estudo deverá responder:

> **Quais informações realmente ajudam a antecipar a volatilidade das commodities estratégicas para a Bahia, em quais horizontes e em quais condições de mercado?**

Essa pergunta fornece uma contribuição metodológica, financeira e regional claramente defensável.
