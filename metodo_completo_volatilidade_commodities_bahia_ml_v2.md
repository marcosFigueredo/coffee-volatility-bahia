# Método Completo

## Previsão de Volatilidade do Café (Arábica e Conilon) na Bahia com Machine Learning

### Título provisório do artigo
**Cross-Market Machine Learning for Coffee Price Volatility Forecasting: Evidence from Arabica and Conilon in Bahia, Brazil**

---

# 1. Visão geral metodológica

O estudo será desenvolvido como um problema supervisionado de previsão de séries temporais financeiras, no qual o objetivo é prever a volatilidade futura de commodities agrícolas estratégicas para a Bahia a partir de três grupos principais de informação:

1. histórico da própria commodity;
2. informações provenientes de outras commodities;
3. variáveis financeiras e macrofinanceiras.

O desenho experimental será baseado em comparação incremental entre conjuntos de variáveis e entre diferentes famílias de modelos, com validação temporal estrita e prevenção de vazamento de informação.


## 1.1 Objeto científico central

O estudo não tratará a volatilidade apenas como uma variável a ser prevista. O objeto científico central será a **previsibilidade incremental da volatilidade local condicionada à informação cross-market**.

Para uma commodity \(c\), em uma praça local da Bahia, definimos três conjuntos de informação disponíveis no instante \(t\):

\[
\mathcal{I}^{L}_{c,t}
\]

como o conjunto de informações locais da própria commodity;

\[
\mathcal{I}^{C}_{t}
\]

como o conjunto de informações provenientes de outras commodities e dos mercados nacional e internacional; e

\[
\mathcal{I}^{F}_{t}
\]

como o conjunto de informações financeiras e macrofinanceiras.

A informação completa disponível é:

\[
\mathcal{I}^{ALL}_{c,t}
=
\mathcal{I}^{L}_{c,t}
\cup
\mathcal{I}^{C}_{t}
\cup
\mathcal{I}^{F}_{t}.
\]

O problema empírico será estimar:

\[
\widehat{V}_{c,t,h}
=
f_h\left(\mathcal{I}_{t}\right)
\]

e verificar quanto o erro de previsão se reduz quando o conjunto de informação passa de:

\[
\mathcal{I}^{L}_{c,t}
\]

para:

\[
\mathcal{I}^{L}_{c,t}\cup\mathcal{I}^{C}_{t},
\]

depois para:

\[
\mathcal{I}^{L}_{c,t}\cup\mathcal{I}^{F}_{t},
\]

e finalmente para:

\[
\mathcal{I}^{ALL}_{c,t}.
\]

Portanto, a contribuição não será formulada como simples comparação entre algoritmos. O interesse central será:

\[
\boxed{
\Delta \text{Predictability}
}
\]

isto é, **o ganho de previsibilidade da volatilidade local produzido pela incorporação de informação externa ao próprio mercado local**.

Essa interpretação é deliberadamente preditiva, e não causal. Uma redução de erro após a inclusão de variáveis externas indicará conteúdo informacional incremental, não uma decomposição estrutural da volatilidade nem causalidade econômica.

O pipeline metodológico será:

```text
Aquisição dos dados
        |
        v
Auditoria e controle de qualidade
        |
        v
Padronização e sincronização temporal
        |
        v
Construção de retornos e volatilidade
        |
        v
Engenharia de atributos
        |
        v
Definição dos regimes de mercado
        |
        v
Separação temporal treino/validação/teste
        |
        v
Treinamento dos modelos
        |
        v
Previsão walk-forward
        |
        v
Avaliação por métricas
        |
        v
Testes estatísticos
        |
        v
Explicabilidade
        |
        v
Análises de robustez
```

---

# 2. Desenho do estudo

O estudo será observacional, quantitativo e longitudinal, baseado em séries temporais históricas.

A unidade temporal principal será o dia útil de negociação. Caso algumas séries locais apresentem baixa densidade ou grande número de dias sem atualização, será considerada uma versão semanal da análise.

> **Nota de escopo (atualizada em 2026-09-09):** a auditoria inicial dos dados (`data_audit.csv`) mostrou que as praças-alvo de soja, algodão e cacau no SEAGRI-BA têm gaps de 400 a mais de 4000 dias, inviabilizando-as como commodity-alvo. Café Arábica e Conilon foram as únicas com continuidade suficiente. O escopo foi restringido a essas duas variedades como alvo; soja, algodão e cacau permanecem no desenho apenas como variáveis explicativas cross-commodity (seção 13.2 do plano geral).

As commodities-alvo (variável dependente) serão:

- café Arábica;
- café Conilon/Robusta.

As commodities cross-market (variáveis explicativas, não são alvo) serão:

- soja;
- algodão;
- cacau.

---

# 3. Fontes de dados

## 3.1 Dados locais da Bahia

As séries locais (alvo) serão obtidas da SEAGRI-BA.

Combinações consideradas:

- café Arábica — Vitória da Conquista;
- café Arábica — Luís Eduardo Magalhães;
- café Conilon — Eunápolis.

Sempre que possível, as séries locais serão utilizadas como variável-alvo principal.

---

## 3.2 Dados nacionais

Serão utilizadas séries do CEPEA como referências nacionais para café Arábica e café Robusta (download automático ainda bloqueado por proteção anti-bot; pendente de obtenção manual).

Essas séries poderão ser utilizadas:

- como benchmark nacional;
- como variável explicativa;
- como substituta da série local em caso de baixa frequência;
- para análise de transmissão entre mercado local e mercado nacional.

---

## 3.3 Dados internacionais

Serão utilizadas séries de referência dos mercados futuros. Café é o alvo internacional/benchmark; soja, cacau e algodão são cross-commodity:

- café — ICE Coffee C (proxy coletado: Yahoo Finance `KC=F`);
- soja — CME Soybean Futures (proxy coletado: Yahoo Finance `ZS=F`);
- cacau — ICE Cocoa Futures (proxy coletado: Yahoo Finance `CC=F`);
- algodão — ICE Cotton No. 2 (proxy coletado: Yahoo Finance `CT=F`).

Quando dados oficiais completos não estiverem disponíveis gratuitamente, serão utilizados contratos contínuos de fontes públicas, documentando-se explicitamente a natureza complementar da série.

---

## 3.4 Variáveis financeiras

Serão consideradas:

- USD/BRL;
- SELIC;
- VIX;
- Brent;
- WTI;
- Dollar Index;
- S&P 500;
- Ibovespa;
- Treasury de 10 anos;
- Federal Funds Rate.

---

# 4. Auditoria inicial dos dados

Antes de qualquer modelagem, todos os arquivos CSV serão submetidos a uma auditoria automatizada.

Para cada série serão calculados:

- fonte;
- commodity;
- praça;
- unidade;
- data inicial;
- data final;
- número de observações;
- proporção de valores ausentes;
- número de duplicatas;
- percentual de observações repetidas;
- maior intervalo sem observação;
- frequência temporal predominante;
- número médio de observações por mês;
- percentual de dias úteis cobertos;
- número de mudanças de preço;
- número de potenciais outliers.

Será produzido o arquivo:

```text
data_audit.csv
```

---

# 5. Critérios de inclusão das séries

Uma série será considerada adequada para modelagem diária quando apresentar:

1. histórico temporal suficientemente longo;
2. frequência compatível com dias úteis;
3. baixa proporção de dados ausentes;
4. poucas lacunas prolongadas;
5. número suficiente de mudanças de preço;
6. ausência de alterações metodológicas severas sem documentação;
7. consistência de unidade ao longo do tempo.

Caso uma série local não satisfaça esses critérios:

- será reamostrada para frequência semanal; ou
- será mantida apenas para validação regional; ou
- será substituída por referência CEPEA/internacional como alvo principal.

---

# 6. Padronização temporal

Todas as séries serão convertidas para um índice temporal comum.

Etapas:

1. converter datas para padrão ISO;
2. remover registros duplicados;
3. ordenar cronologicamente;
4. identificar feriados e dias sem negociação;
5. alinhar as séries por interseção temporal;
6. preservar somente informações disponíveis até o instante \(t\).

Não será realizado preenchimento futuro.

---

# 7. Tratamento de dados ausentes

A estratégia dependerá do tipo de variável.

## 7.1 Preços da commodity-alvo

Não será utilizada interpolação linear entre preços de mercado quando isso puder gerar retornos artificiais.

Preferências:

- manter dias efetivamente observados;
- trabalhar com interseção de datas;
- reamostrar semanalmente, quando necessário.

## 7.2 Variáveis financeiras

Para variáveis macrofinanceiras com baixa frequência de atualização, poderá ser utilizado forward-fill apenas quando economicamente justificável.

Exemplo:

- taxas de juros;
- indicadores macroeconômicos.

O procedimento será aplicado exclusivamente usando informações disponíveis até \(t\).

---

# 8. Tratamento de outliers

Outliers não serão removidos automaticamente, pois movimentos extremos são parte importante da volatilidade financeira.

Serão analisados:

- erros evidentes de registro;
- preços negativos;
- zeros incompatíveis;
- saltos associados a mudança de unidade;
- valores fora de limites fisicamente ou economicamente plausíveis.

Eventos extremos reais serão preservados.

---

# 9. Padronização de unidades

Cada série será convertida, quando necessário, para unidade consistente ao longo de todo o período.

Serão documentadas:

- unidade original;
- fator de conversão;
- data de eventual mudança de unidade;
- unidade final adotada.

Os modelos serão preferencialmente treinados sobre retornos, reduzindo problemas associados a diferenças de escala de preços.

---

# 10. Construção dos retornos

Para uma commodity \(c\), com preço observado \(P_{c,t}>0\), será calculado o retorno logarítmico diário:

\[
r_{c,t}
=
100\ln\left(
\frac{P_{c,t}}{P_{c,t-1}}
\right).
\]

O retorno é a variável básica a partir da qual o risco de preço será construído. A utilização de retornos, em vez de preços em nível, reduz problemas de escala e permite comparar mercados com unidades monetárias distintas.

---

# 11. Conceito de volatilidade adotado no estudo

## 11.1 Volatilidade como intensidade de variação

Em termos populacionais, a volatilidade de um retorno futuro pode ser representada pela variância condicional:

\[
\sigma^2_{c,t+h\mid\mathcal{I}_t}
=
\operatorname{Var}
\left(
r_{c,t+h}
\mid
\mathcal{I}_t
\right)
\]

e, em unidades de desvio-padrão,

\[
\sigma_{c,t+h\mid\mathcal{I}_t}
=
\sqrt{
\sigma^2_{c,t+h\mid\mathcal{I}_t}
}.
\]

Essa quantidade é latente e não é observada diretamente. O estudo utilizará uma medida observável de variação futura construída a partir dos retornos realizados.

## 11.2 Variação futura agregada

Para um horizonte de \(h\) dias, será definida a **variação quadrática futura agregada**:

\[
Q_{c,t,h}
=
\sum_{i=1}^{h}
r_{c,t+i}^{2}.
\]

A medida correspondente em escala de volatilidade será:

\[
V_{c,t,h}
=
\sqrt{
Q_{c,t,h}
}
=
\sqrt{
\sum_{i=1}^{h}
r_{c,t+i}^{2}
}.
\]

Assim:

- \(Q_{c,t,h}\) será o objeto em escala de variância;
- \(V_{c,t,h}\) será o objeto em escala de volatilidade.

Serão considerados inicialmente:

\[
h\in\{1,5,20\}.
\]

A ênfase principal será dada a \(h=5\) e \(h=20\), aproximadamente correspondentes a uma semana e um mês de negociação.

## 11.3 Distinção em relação à realized volatility intradiária

O termo *realized volatility* é frequentemente utilizado quando a variação diária é estimada pela soma de retornos intradiários:

\[
RV^{intra}_{c,t}
=
\sum_{j=1}^{m}
r_{c,t,j}^{2},
\]

onde \(j\) representa observações intradiárias.

Como as séries locais e nacionais deste estudo são predominantemente diárias, não será assumido que possuímos uma medida intradiária clássica de *realized volatility*.

A terminologia principal será:

- **forward aggregated squared-return variation** para \(Q_{c,t,h}\);
- **forward aggregated return volatility** para \(V_{c,t,h}\).

Em português:

- variação quadrática futura agregada;
- volatilidade futura agregada dos retornos.

Essa distinção evita atribuir aos dados diários uma granularidade que eles não possuem.

---

# 12. Volatilidade histórica disponível em \(t\)

Além do alvo futuro, serão construídas medidas históricas causais usando apenas observações disponíveis até \(t\).

Para uma janela passada de \(k\) dias:

\[
HVar_{c,t,k}
=
\sum_{i=0}^{k-1}
r_{c,t-i}^{2}
\]

e:

\[
HVol_{c,t,k}
=
\sqrt{
HVar_{c,t,k}
}.
\]

Também poderá ser utilizada a forma baseada no desvio-padrão amostral:

\[
s_{c,t,k}
=
\sqrt{
\frac{1}{k-1}
\sum_{i=0}^{k-1}
\left(
r_{c,t-i}
-
\bar r_{c,t,k}
\right)^2
}.
\]

Serão consideradas inicialmente:

\[
k\in\{5,10,20,60\}.
\]

Essas medidas pertencem a \(\mathcal{I}^{L}_{c,t}\) e podem ser utilizadas como preditoras, pois dependem apenas do passado.

---

# 13. Variável-alvo e representação estatística

## 13.1 Alvo principal

O alvo primário para modelagem será:

\[
Q_{c,t,h}
=
\sum_{i=1}^{h}
r_{c,t+i}^{2}.
\]

Essa escolha é conveniente porque:

1. \(Q_{c,t,h}\ge 0\);
2. está diretamente ligado à intensidade acumulada das variações futuras;
3. é compatível com funções de perda para previsão de variância, especialmente QLIKE;
4. permite recuperar a volatilidade interpretável por:

\[
V_{c,t,h}
=
\sqrt{
Q_{c,t,h}
}.
\]

## 13.2 Transformação logarítmica

Como \(Q_{c,t,h}\) tende a apresentar assimetria positiva, será também avaliada:

\[
Y_{c,t,h}
=
\log
\left(
Q_{c,t,h}+\epsilon
\right),
\]

com \(\epsilon>0\) pequeno.

Se o modelo for treinado na escala logarítmica, a previsão será transformada de volta para a escala original antes do cálculo das métricas de variância.

## 13.3 Restrição de positividade

Como previsões de variância devem ser não negativas, será garantido:

\[
\widehat{Q}_{c,t,h}>0.
\]

Para modelos capazes de produzir valores negativos, poderá ser utilizada transformação logarítmica ou truncamento numérico no limite mínimo:

\[
\widehat{Q}^{+}_{c,t,h}
=
\max
\left(
\widehat{Q}_{c,t,h},
\epsilon
\right).
\]

---

# 13A. Estrutura matemática da previsibilidade incremental

Para cada commodity \(c\) e horizonte \(h\), serão construídos quatro estimadores aninhados.

## 13A.1 Informação local

\[
\widehat{Q}^{L}_{c,t,h}
=
f^{L}_{c,h}
\left(
\mathcal{I}^{L}_{c,t}
\right).
\]

## 13A.2 Informação local + cross-commodity

\[
\widehat{Q}^{LC}_{c,t,h}
=
f^{LC}_{c,h}
\left(
\mathcal{I}^{L}_{c,t}
\cup
\mathcal{I}^{C}_{t}
\right).
\]

## 13A.3 Informação local + financeira

\[
\widehat{Q}^{LF}_{c,t,h}
=
f^{LF}_{c,h}
\left(
\mathcal{I}^{L}_{c,t}
\cup
\mathcal{I}^{F}_{t}
\right).
\]

## 13A.4 Informação completa

\[
\widehat{Q}^{LCF}_{c,t,h}
=
f^{LCF}_{c,h}
\left(
\mathcal{I}^{L}_{c,t}
\cup
\mathcal{I}^{C}_{t}
\cup
\mathcal{I}^{F}_{t}
\right).
\]

Seja:

\[
\mathcal{L}
\left(
Q,\widehat Q
\right)
\]

uma função de perda out-of-sample. O risco preditivo de um conjunto de informação \(S\) será estimado por:

\[
R_S
=
\mathbb{E}
\left[
\mathcal{L}
\left(
Q_{c,t,h},
\widehat{Q}^{S}_{c,t,h}
\right)
\right].
\]

A previsibilidade incremental cross-commodity será:

\[
G_C
=
\frac{
R_L-R_{LC}
}{
R_L
}.
\]

A contribuição preditiva financeira será:

\[
G_F
=
\frac{
R_L-R_{LF}
}{
R_L
}.
\]

O ganho total de informação externa será:

\[
G_{CF}
=
\frac{
R_L-R_{LCF}
}{
R_L
}.
\]

A interpretação será:

\[
G>0
\Rightarrow
\text{ganho preditivo},
\]

\[
G=0
\Rightarrow
\text{ausência de ganho},
\]

\[
G<0
\Rightarrow
\text{a informação adicional piora a previsão out-of-sample}.
\]

Essa possibilidade de \(G<0\) é metodologicamente importante: mais informação não implica necessariamente melhor previsão em amostras finitas.

## 13A.5 Complementaridade entre informação cross-commodity e financeira

Será também calculado:

\[
G_{C\mid F}
=
\frac{
R_{LF}-R_{LCF}
}{
R_{LF}
}
\]

para medir o ganho de adicionar commodities quando as informações financeiras já estão presentes, e:

\[
G_{F\mid C}
=
\frac{
R_{LC}-R_{LCF}
}{
R_{LC}
}
\]

para medir o ganho de adicionar informação financeira quando as informações cross-commodity já estão presentes.

Essas quantidades permitem distinguir:

- contribuição marginal cross-commodity;
- contribuição marginal financeira;
- complementaridade entre os dois blocos.

## 13A.6 Ganho condicionado ao regime de mercado

Para um regime \(z\in\{Low,Normal,High\}\):

\[
R_{S}^{(z)}
=
\mathbb{E}
\left[
\mathcal{L}
\left(
Q,\widehat Q^{S}
\right)
\mid
Z_t=z
\right].
\]

Consequentemente:

\[
G_{CF}^{(z)}
=
\frac{
R_L^{(z)}-R_{LCF}^{(z)}
}{
R_L^{(z)}
}.
\]

Isso permitirá testar diretamente se a informação externa possui maior valor preditivo em períodos turbulentos:

\[
G_{CF}^{(High)}
>
G_{CF}^{(Normal)}
\]

e:

\[
G_{CF}^{(High)}
>
G_{CF}^{(Low)}.
\]

## 13A.7 Interpretação econômica

A estrutura acima não decompõe matematicamente a volatilidade observada em uma parcela “local” e uma parcela “externa”. O que será decomposto é **a capacidade de previsão**.

Portanto:

\[
\text{volatilidade observada}
\neq
\text{contribuição preditiva}.
\]

O objeto mensurado será:

\[
\boxed{
\text{valor informacional incremental para previsão}
}
\]

e não uma fração causal da volatilidade atribuível a cada mercado.

---

# 14. Engenharia de atributos

## 14.1 Atributos da própria commodity

Para cada commodity-alvo serão calculados:

- retornos defasados;
- retorno absoluto;
- retorno ao quadrado;
- volatilidade de 5 dias;
- volatilidade de 10 dias;
- volatilidade de 20 dias;
- volatilidade de 60 dias;
- média móvel de preços;
- média móvel de retornos;
- desvio-padrão móvel;
- mínimo móvel;
- máximo móvel;
- amplitude móvel;
- momentum;
- drawdown;
- skewness móvel;
- kurtosis móvel.

---

# 15. Atributos de mercado futuro

Quando disponíveis:

- preço de fechamento;
- retorno;
- volume;
- open interest;
- variação do volume;
- variação do open interest;
- amplitude high-low;
- diferença entre mercado local e futuro.

---

# 16. Basis

Quando houver preços comparáveis entre mercado físico e mercado futuro, será calculado:

\[
Basis_t = S_t - F_t
\]

onde:

- \(S_t\): preço spot;
- \(F_t\): preço futuro.

Quando necessário, os preços serão previamente convertidos para a mesma moeda e unidade.

---

# 17. Atributos cross-commodity

Para uma commodity-alvo \(c\), serão utilizadas informações das demais commodities \(j\neq c\).

Exemplos:

\[
r_{j,t}
\]

\[
|r_{j,t}|
\]

\[
RV_{j,t,5}
\]

\[
RV_{j,t,20}
\]

Também poderão ser incluídas correlações móveis entre commodities.

---

# 18. Atributos financeiros

Serão consideradas:

## Câmbio

- USD/BRL;
- retorno do USD/BRL;
- volatilidade do USD/BRL;
- médias móveis do USD/BRL.

## Risco internacional

- VIX;
- variação diária do VIX;
- média móvel do VIX;
- regime de VIX.

## Energia

- Brent;
- retorno do Brent;
- volatilidade do Brent;
- WTI;
- retorno do WTI.

## Mercados acionários

- S&P 500;
- Ibovespa;
- retornos e volatilidades.

## Juros

- SELIC;
- Fed Funds;
- Treasury 10 anos.

---

# 19. Defasagens

Serão usadas apenas defasagens passadas.

Conjunto inicial:

\[
L = \{1,2,3,5,10,20\}
\]

Exemplo:

\[
X_t=
\{
r_{t-1},
r_{t-2},
RV_{t-5},
VIX_{t-1},
USD_{t-1}
\}
\]

Nenhuma variável em \(t+h\) será usada como preditora.

---

# 20. Seleção de atributos

A seleção de atributos será realizada somente dentro do conjunto de treinamento.

Serão consideradas:

- remoção de features com variância quase nula;
- remoção de variáveis altamente redundantes;
- importância por ganho em modelos de boosting;
- permutation importance;
- SHAP.

Não será utilizada seleção baseada no conjunto de teste.

---

# 21. Normalização

Modelos baseados em árvores não necessitam de normalização obrigatória.

Para LSTM e TCN será utilizada padronização:

\[
z = \frac{x-\mu_{train}}{\sigma_{train}}
\]

Os parâmetros \(\mu\) e \(\sigma\) serão calculados exclusivamente no conjunto de treinamento de cada janela temporal.

---

# 22. Definição de regimes de volatilidade

O mercado será classificado inicialmente em três regimes.

Para uma medida de volatilidade \(V_t\):

\[
R_t=
\begin{cases}
Low,&V_t<Q_{0.33}\\
Normal,&Q_{0.33}\le V_t\le Q_{0.66}\\
High,&V_t>Q_{0.66}
\end{cases}
\]

Os quantis serão calculados somente sobre dados passados no respectivo período de treinamento.

---

# 23. Análise alternativa de regimes

Como análise de robustez, poderá ser aplicado Hidden Markov Model.

Nesse caso, será avaliada solução com:

- 2 estados;
- 3 estados.

A quantidade final será escolhida com base em:

- AIC;
- BIC;
- estabilidade dos estados;
- interpretabilidade econômica.

---

# 24. Baselines ingênuos

Antes dos modelos econométricos e de ML serão implementados modelos ingênuos.

## 24.1 Persistence

\[
\hat{RV}_{t+h}=RV_t
\]

## 24.2 Historical mean

\[
\hat{RV}_{t+h}
=
\frac{1}{k}
\sum_{i=0}^{k-1}RV_{t-i}
\]

Esses modelos garantem que ganhos complexos sejam comparados contra referências mínimas.

---

# 25. Modelos econométricos

Serão avaliados:

## 25.1 GARCH(1,1)

\[
\sigma_t^2
=
\omega+
\alpha\epsilon_{t-1}^{2}
+
\beta\sigma_{t-1}^{2}
\]

## 25.2 EGARCH

Modelo apropriado para assimetria entre choques positivos e negativos.

## 25.3 HAR-RV

\[
RV_{t+1}
=
\beta_0
+
\beta_d RV_t
+
\beta_w RV_t^{(5)}
+
\beta_m RV_t^{(22)}
+
\epsilon_t
\]

O HAR-RV será usado quando a construção da volatilidade for compatível com sua interpretação.

---

# 26. Modelos de Machine Learning

## 26.1 Random Forest

Utilizado como baseline não linear.

Hiperparâmetros principais:

- número de árvores;
- profundidade máxima;
- mínimo de observações por folha;
- número de features por divisão.

## 26.2 XGBoost

Hiperparâmetros principais:

- learning rate;
- max depth;
- subsample;
- colsample;
- número de estimadores;
- regularização L1/L2.

## 26.3 LightGBM

Modelo principal tabular.

Hiperparâmetros:

- num_leaves;
- learning_rate;
- n_estimators;
- max_depth;
- feature_fraction;
- bagging_fraction;
- lambda_l1;
- lambda_l2.

---

# 27. Modelos de Deep Learning

## 27.1 LSTM

Entrada:

\[
X_{t-L+1:t}
\]

com comprimento de sequência \(L\).

Valores iniciais:

\[
L \in \{20,40,60\}
\]

Arquitetura inicial:

- 1 ou 2 camadas LSTM;
- dropout;
- camada fully connected;
- saída contínua.

## 27.2 TCN

Arquitetura composta por:

- convoluções causais;
- dilatações crescentes;
- blocos residuais;
- dropout;
- camada linear final.

A TCN será comparada diretamente à LSTM.

---

# 28. Função de perda

Para ML tabular:

- MSE;
- MAE;
- Huber, em análise de robustez.

Para Deep Learning:

- MSE;
- MAE;
- eventualmente perda baseada em QLIKE.

---

# 29. Estratégia experimental

Os experimentos serão definidos como conjuntos de informação aninhados.

## E0 — Baselines

\[
\text{Naive + GARCH + EGARCH + HAR}
\]

O objetivo de E0 será estabelecer a referência mínima de desempenho.

## E1 — Local / Own history

\[
\mathcal{I}_{E1}
=
\mathcal{I}^{L}_{c,t}
\]

e:

\[
\widehat Q^{E1}_{c,t,h}
=
f
\left(
\mathcal{I}^{L}_{c,t}
\right).
\]

E1 mede quanta informação preditiva está contida no próprio histórico local.

## E2 — Local + Cross-commodity

\[
\mathcal{I}_{E2}
=
\mathcal{I}^{L}_{c,t}
\cup
\mathcal{I}^{C}_{t}
\]

e:

\[
\widehat Q^{E2}_{c,t,h}
=
f
\left(
\mathcal{I}^{L}_{c,t},
\mathcal{I}^{C}_{t}
\right).
\]

A comparação E2 versus E1 mede o conteúdo informacional incremental dos demais mercados de commodities.

## E3 — Local + Financial

\[
\mathcal{I}_{E3}
=
\mathcal{I}^{L}_{c,t}
\cup
\mathcal{I}^{F}_{t}
\]

e:

\[
\widehat Q^{E3}_{c,t,h}
=
f
\left(
\mathcal{I}^{L}_{c,t},
\mathcal{I}^{F}_{t}
\right).
\]

A comparação E3 versus E1 mede o conteúdo informacional incremental do ambiente financeiro.

## E4 — Full cross-market information

\[
\mathcal{I}_{E4}
=
\mathcal{I}^{L}_{c,t}
\cup
\mathcal{I}^{C}_{t}
\cup
\mathcal{I}^{F}_{t}
\]

e:

\[
\widehat Q^{E4}_{c,t,h}
=
f
\left(
\mathcal{I}^{L}_{c,t},
\mathcal{I}^{C}_{t},
\mathcal{I}^{F}_{t}
\right).
\]

E4 representa a hipótese de informação integrada.

---

# 30. Quantificação do ganho de previsibilidade

Se \(R_{E1},R_{E2},R_{E3},R_{E4}\) forem os riscos out-of-sample medidos pela mesma função de perda, serão calculados:

## 30.1 Cross-Market Predictive Gain

\[
CMPG_C
=
100
\left(
1-
\frac{R_{E2}}{R_{E1}}
\right).
\]

## 30.2 Financial Predictive Gain

\[
FPG
=
100
\left(
1-
\frac{R_{E3}}{R_{E1}}
\right).
\]

## 30.3 Integrated Predictive Gain

\[
IPG
=
100
\left(
1-
\frac{R_{E4}}{R_{E1}}
\right).
\]

Os indicadores serão expressos em porcentagem.

Exemplo:

\[
IPG=12\%
\]

significará que o modelo completo reduziu em 12% a perda out-of-sample em relação ao modelo baseado exclusivamente em informação local.

Também serão calculados ganhos condicionais:

\[
CMPG_{C\mid F}
=
100
\left(
1-
\frac{R_{E4}}{R_{E3}}
\right)
\]

e:

\[
FPG_{F\mid C}
=
100
\left(
1-
\frac{R_{E4}}{R_{E2}}
\right).
\]

Esses indicadores serão utilizados como medidas operacionais de **previsibilidade incremental**, não como estimativas causais.

---

# 31. Divisão temporal dos dados

Não será utilizado split aleatório.

Será adotado esquema walk-forward.

Exemplo:

```text
Treino: 2010–2018 | validação: 2019 | teste: 2020
Treino: 2010–2019 | validação: 2020 | teste: 2021
Treino: 2010–2020 | validação: 2021 | teste: 2022
...
```

A configuração real dependerá do início das séries.

---

# 32. Expanding window

Na configuração principal, será utilizado expanding window.

Para iteração \(k\):

\[
Train_k=[t_0,t_k]
\]

\[
Test_k=[t_k+1,t_k+h]
\]

O conjunto de treinamento aumenta progressivamente com novas observações.

---

# 33. Rolling window

Como análise de robustez, será utilizado rolling window.

Nesse caso:

\[
Train_k=[t_k-W,t_k]
\]

onde \(W\) é o comprimento fixo da janela.

Serão testados valores compatíveis com:

- 2 anos;
- 3 anos;
- 5 anos.

---

# 34. Validação interna

O tuning de hiperparâmetros deverá ocorrer apenas no conjunto de treinamento.

Será utilizada validação temporal interna:

```text
Train 1 -> Validation 1
Train 2 -> Validation 2
Train 3 -> Validation 3
```

Não será utilizada K-fold aleatória.

---

# 35. Otimização de hiperparâmetros

Será utilizada preferencialmente busca Bayesiana ou Optuna.

A função objetivo será baseada em:

- QLIKE; ou
- MAE.

O número de trials será limitado para evitar otimização excessiva.

Sugestão inicial:

- 30–50 trials para modelos tabulares;
- 20–30 trials para LSTM/TCN.

---

# 36. Controle de overfitting

Serão utilizados:

- regularização;
- early stopping;
- limitação da profundidade;
- dropout;
- validação temporal;
- redução do número de features;
- comparação treino versus validação.

---

# 37. Métricas principais

As métricas serão calculadas na escala apropriada para cada interpretação.

## 37.1 QLIKE

Para previsão da variação quadrática \(Q_{c,t,h}\), será utilizada:

\[
QLIKE
=
\frac{1}{n}
\sum_{t=1}^{n}
\left[
\log
\left(
\widehat Q_t
\right)
+
\frac{Q_t}{\widehat Q_t}
\right].
\]

Uma forma equivalente, que difere apenas por termos independentes do modelo, é:

\[
QLIKE^{*}
=
\frac{1}{n}
\sum_{t=1}^{n}
\left[
\frac{Q_t}{\widehat Q_t}
-
\log
\left(
\frac{Q_t}{\widehat Q_t}
\right)
-
1
\right].
\]

Menor valor indica melhor previsão.

O QLIKE será calculado sobre:

\[
Q_t
\]

e não diretamente sobre:

\[
V_t=\sqrt{Q_t},
\]

preservando sua interpretação como função de perda para previsão de variância.

## 37.2 MAE da volatilidade

Para interpretação em escala de volatilidade:

\[
MAE_V
=
\frac{1}{n}
\sum_{t=1}^{n}
\left|
V_t-\widehat V_t
\right|,
\]

onde:

\[
\widehat V_t
=
\sqrt{\widehat Q_t}.
\]

## 37.3 RMSE da volatilidade

\[
RMSE_V
=
\sqrt{
\frac{1}{n}
\sum_{t=1}^{n}
\left(
V_t-\widehat V_t
\right)^2
}.
\]

Também poderão ser reportados MAE e RMSE em \(Q\), mas QLIKE será a métrica principal para comparação da qualidade das previsões de variância.

---

# 38. Métricas relativas

Também serão calculadas razões de erro em relação ao baseline.

\[
RelativeError
=
\frac{Error(Model)}
{Error(Baseline)}
\]

Valores menores que 1 indicam melhoria.

---

# 39. Teste de Diebold-Mariano

Para dois modelos concorrentes \(A\) e \(B\):

\[
d_t
=
L(e_{A,t})-L(e_{B,t})
\]

Será testada:

\[
H_0:E[d_t]=0
\]

contra:

\[
H_1:E[d_t]\neq0
\]

Será utilizada correção adequada para autocorrelação quando \(h>1\).

---

# 40. Comparação múltipla

Como vários modelos serão comparados, será reportado:

- p-valor bruto;
- p-valor corrigido por Holm, quando aplicável.

Isso reduzirá risco de conclusões espúrias por múltiplos testes.

---

# 41. Avaliação por regime

As métricas serão calculadas separadamente para:

- low volatility;
- normal volatility;
- high volatility.

Isso permitirá avaliar se um modelo aparentemente superior globalmente mantém desempenho em períodos extremos.

---

# 42. Avaliação por horizonte

Todos os resultados serão separados para:

\[
h=1
\]

\[
h=5
\]

\[
h=20
\]

A comparação entre horizontes será uma dimensão central do estudo.

---

# 43. Avaliação por variedade

Serão reportadas métricas individuais para:

- café Arábica;
- café Conilon.

Também será calculado desempenho médio entre as duas variedades, sem ocultar heterogeneidade individual.

---

# 44. Explicabilidade com SHAP

Para o melhor modelo tabular será aplicado SHAP.

Serão produzidos:

- SHAP summary plot;
- ranking global;
- dependência das principais variáveis;
- importância por commodity;
- importância por regime.

---

# 45. Importância por grupo de variáveis

As features serão agrupadas em:

```text
OWN
CROSS-COMMODITY
FINANCIAL
FUTURES
MACRO
```

Será calculada a contribuição agregada de cada grupo.

Isso permitirá responder diretamente:

> qual bloco de informação realmente sustenta a previsão?

---

# 46. Análise temporal da importância

A importância poderá ser calculada por períodos.

Exemplo:

- período pré-2020;
- 2020–2021;
- 2022–2023;
- período recente.

O objetivo será avaliar estabilidade estrutural.

---

# 47. Correlações e dependência

Antes da modelagem serão calculadas:

- Pearson;
- Spearman;
- correlação móvel.

Esses resultados terão caráter descritivo, não causal.

---

# 48. Análise de causalidade preditiva

Como análise adicional, poderá ser utilizado Granger causality entre séries selecionadas.

Essa análise será tratada como evidência de precedência temporal preditiva, não como causalidade econômica estrutural.

---

# 49. Robustez I — Frequência temporal

Se os dados permitirem, serão comparados:

- diário;
- semanal.

O objetivo será verificar se a baixa atualização de preços locais interfere nos resultados.

---

# 50. Robustez II — Fontes

Serão comparados:

- somente fontes oficiais;
- fontes oficiais + complementares.

Isso permitirá verificar dependência excessiva de séries de provedores secundários.

---

# 51. Robustez III — Períodos extremos

Serão realizadas análises:

- com todos os dados;
- sem os maiores 1% eventos extremos;
- apenas nos maiores 10% episódios de volatilidade.

---

# 52. Robustez IV — Janela temporal

Serão comparados:

- expanding window;
- rolling window.

---

# 53. Robustez V — Métrica de volatilidade

Quando os dados permitirem, poderão ser comparadas:

- realized volatility;
- desvio-padrão móvel;
- Parkinson volatility;
- Garman-Klass volatility.

As duas últimas somente serão utilizadas quando existirem preços OHLC confiáveis.

---

# 54. Robustez VI — Transformação do alvo

Serão comparados:

- volatilidade em nível;
- log-volatilidade.

---

# 55. Robustez VII — Modelos reduzidos

Será analisado se modelos simples com poucas variáveis podem alcançar desempenho semelhante aos modelos completos.

Isso é relevante para interpretação e eventual aplicação operacional.

---

# 56. Ausência de vazamento temporal

Serão explicitamente proibidos:

- normalização usando toda a série;
- imputação com dados futuros;
- seleção de atributos antes do split;
- cálculo de quantis de regime usando dados futuros;
- tuning sobre o conjunto de teste;
- cálculo de médias móveis centradas.

Todas as janelas serão causais.

---

# 57. Reprodutibilidade

O projeto deverá registrar:

- seed aleatória;
- versão das bibliotecas;
- período de coleta;
- fonte dos dados;
- parâmetros dos modelos;
- hiperparâmetros finais;
- configuração das janelas;
- datas de cada fold.

---

# 58. Organização dos arquivos

Estrutura recomendada:

```text
project/
|
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
|
├── src/
│   ├── 01_audit_data.py
│   ├── 02_clean_data.py
│   ├── 03_build_features.py
│   ├── 04_build_targets.py
│   ├── 05_build_folds.py
│   ├── 06_train_baselines.py
│   ├── 07_train_ml.py
│   ├── 08_train_dl.py
│   ├── 09_evaluate.py
│   ├── 10_dm_tests.py
│   └── 11_shap.py
|
├── results/
│   ├── metrics/
│   ├── predictions/
│   ├── models/
│   ├── tables/
│   └── figures/
|
├── config/
│   └── config.yaml
|
└── README.md
```

---

# 59. Saídas mínimas por modelo

Cada execução deverá produzir:

```text
commodity
horizon
model
experiment
fold
date
y_true
y_pred
regime
```

Arquivo sugerido:

```text
predictions_long.csv
```

Essa estrutura permitirá recalcular qualquer métrica posteriormente.

---

# 60. Arquivo de métricas

Gerar:

```text
metrics.csv
```

com:

```text
commodity
horizon
model
experiment
fold
regime
QLIKE
MAE
RMSE
```

---

# 61. Arquivo de hiperparâmetros

Gerar:

```text
best_params.csv
```

contendo:

```text
commodity
horizon
model
fold
parameter
value
```

---

# 62. Arquivo de testes estatísticos

Gerar:

```text
dm_tests.csv
```

com:

```text
commodity
horizon
model_a
model_b
loss
dm_stat
p_value
p_adjusted
significant
```

---

# 63. Critério de modelo principal

O modelo principal não será escolhido apenas pelo menor RMSE.

Será utilizado critério hierárquico:

1. menor QLIKE médio;
2. estabilidade entre folds;
3. significância versus baselines;
4. desempenho em high-volatility regime;
5. consistência entre commodities;
6. interpretabilidade.

---

# 64. Critério de evidência para H1

H1 será suportada se modelos de ML apresentarem:

- menor erro médio;
- vantagem consistente entre folds;
- resultado significativo versus GARCH/HAR em parte substancial dos experimentos.

---

# 65. Critério de evidência para H2

H2 será suportada se:

\[
R_{E2}<R_{E1}
\]

e, consequentemente:

\[
CMPG_C>0,
\]

de forma consistente entre folds e com evidência estatística favorável no teste de Diebold-Mariano.

O tamanho do efeito será reportado pelo próprio \(CMPG_C\), evitando limitar a conclusão ao p-valor.

---

# 66. Critério de evidência para H3

H3 será suportada se:

\[
R_{E3}<R_{E1}
\]

e/ou:

\[
R_{E4}<R_{E2}.
\]

Equivalentemente, espera-se:

\[
FPG>0
\]

e/ou:

\[
FPG_{F\mid C}>0.
\]

A interpretação será de valor informacional incremental do bloco financeiro.

---

# 67. Critério de evidência para H4

H4 será suportada se os ganhos condicionados ao regime satisfizerem, de forma consistente:

\[
IPG^{(High)}
>
IPG^{(Normal)}
\]

e:

\[
IPG^{(High)}
>
IPG^{(Low)}.
\]

Também serão examinados separadamente:

\[
CMPG_C^{(High)}
\]

e:

\[
FPG^{(High)}.
\]

Assim, H4 não será avaliada apenas pelo erro absoluto em períodos turbulentos, mas pelo **valor adicional da informação externa nesses períodos**.

---

# 68. Critério de evidência para H5

H5 será suportada se:

1. os rankings de SHAP diferirem entre commodities;
2. a contribuição agregada dos blocos \(\mathcal{I}^{C}\) e \(\mathcal{I}^{F}\) variar entre mercados;
3. os ganhos \(CMPG_C\), \(FPG\) e \(IPG\) apresentarem heterogeneidade entre commodities ou regimes.

Dessa forma, a heterogeneidade será demonstrada tanto por explicabilidade quanto por desempenho preditivo.

---

# 69. Resultados esperados

Espera-se encontrar:

- melhor desempenho de modelos não lineares;
- importância relevante do USD/BRL;
- maior contribuição do VIX em períodos de estresse;
- relações cross-commodity específicas;
- heterogeneidade entre commodities;
- maior ganho do modelo completo em períodos de alta volatilidade.

Esses resultados são hipóteses empíricas e não pressupostos do estudo.

---

# 70. Limitações metodológicas previstas

Possíveis limitações:

- baixa frequência de atualização de preços locais;
- heterogeneidade entre fontes;
- contratos futuros contínuos construídos por provedores secundários;
- ausência de volume/open interest para algumas séries;
- mudanças estruturais ao longo do período;
- diferenças entre mercado físico local e mercado financeiro internacional.

Essas limitações deverão ser explicitadas e testadas por análises de robustez.

---

# 71. Critério de conclusão do método

O pipeline será considerado concluído quando:

1. todas as séries forem auditadas;
2. os alvos forem construídos sem leakage;
3. todos os folds temporais estiverem definidos;
4. E0–E4 forem executados;
5. as previsões forem armazenadas;
6. métricas forem calculadas;
7. testes estatísticos forem executados;
8. análises por horizonte e regime forem concluídas;
9. SHAP for aplicado aos melhores modelos;
10. análises de robustez forem finalizadas.

---

# 72. Sequência computacional recomendada

```text
01_audit_data.py
        |
02_clean_data.py
        |
03_build_features.py
        |
04_build_targets.py
        |
05_build_folds.py
        |
06_train_baselines.py
        |
07_train_ml.py
        |
08_train_dl.py
        |
09_evaluate.py
        |
10_dm_tests.py
        |
11_shap.py
        |
12_robustness.py
        |
13_make_tables.py
        |
14_make_figures.py
```

---

# 73. Método central resumido

Para cada commodity \(c\), praça local e horizonte \(h\), o objeto observado será:

\[
Q_{c,t,h}
=
\sum_{i=1}^{h}
r_{c,t+i}^{2},
\]

com volatilidade correspondente:

\[
V_{c,t,h}
=
\sqrt{Q_{c,t,h}}.
\]

A previsão local básica será:

\[
\widehat Q^{L}_{c,t,h}
=
f
\left(
\mathcal{I}^{L}_{c,t}
\right).
\]

A previsão integrada será:

\[
\widehat Q^{LCF}_{c,t,h}
=
f
\left(
\mathcal{I}^{L}_{c,t},
\mathcal{I}^{C}_{t},
\mathcal{I}^{F}_{t}
\right).
\]

A quantidade científica central será a redução relativa da perda:

\[
\Delta Predictability
=
1-
\frac{
R_{LCF}
}{
R_L
}.
\]

Em termos percentuais:

\[
IPG
=
100
\left(
1-
\frac{
R_{LCF}
}{
R_L
}
\right).
\]

O método, portanto, separa três questões:

\[
\boxed{
\text{quanto o próprio mercado local consegue prever}
}
\]

\[
\boxed{
\text{quanto as commodities externas acrescentam}
}
\]

\[
\boxed{
\text{quanto o sistema financeiro acrescenta}
}
\]

sempre em avaliação estritamente out-of-sample.

A sequência conceitual é:

\[
\mathcal{I}^{L}
\rightarrow
\mathcal{I}^{L}\cup\mathcal{I}^{C}
\rightarrow
\mathcal{I}^{L}\cup\mathcal{I}^{F}
\rightarrow
\mathcal{I}^{L}\cup\mathcal{I}^{C}\cup\mathcal{I}^{F}.
\]

Essa arquitetura transforma o estudo de uma simples comparação de algoritmos em uma análise quantitativa da **origem informacional da previsibilidade da volatilidade local**.

---

# 74. Pergunta final respondida pelo método

O método deverá permitir responder de forma quantitativa:

> **Quanto da previsibilidade out-of-sample da volatilidade futura de commodities estratégicas para a Bahia é explicável pelo histórico local e quanto de ganho preditivo adicional é obtido ao incorporar informações de outros mercados agrícolas e do sistema financeiro?**

