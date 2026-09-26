# Validação temporal e comparação dos modelos de café

## Protocolo corrigido

A previsão é emitida após a observação de t e tem como alvo a raiz da soma dos quadrados dos próximos h retornos, com h em {1, 5, 20}. O horizonte conta observações da praça, não dias corridos nem necessariamente pregões consecutivos. O treinamento é atualizado anualmente, antes de 1º de janeiro do ano de teste.

Para HAR-RV e os modelos de ML, uma linha só entra no treinamento quando a data da última observação usada em seu alvo é anterior ao corte anual. A verificação é feita com `date.shift(-h)`, calculado antes de remover linhas sem alvo. Isso também cobre calendários irregulares e intervalos sem cotação. Para GARCH/EGARCH, os parâmetros são estimados com os retornos disponíveis antes do corte; a recursão é atualizada com os retornos observados até a origem t.

Foram corrigidos três problemas:

1. Alvos de treinamento que terminavam no ano de teste: exclusão de 63, 315 e 1.200 linhas, somadas entre janelas anuais, para h=1, h=5 e h=20, respectivamente. São exclusões por janela de treinamento, não uma redução permanente da base.
2. Origem GARCH/EGARCH em t−1: alterada para t, alinhando a previsão ao alvo futuro utilizado por HAR e ML.
3. Soma de variâncias com valores ausentes: previsões incompletas ou não finitas agora são registradas como falhas, em vez de serem interpretadas como zero ou como uma soma parcial.

A simulação do EGARCH usa semente 42. Os hiperparâmetros dos modelos de ML e o piso QLIKE de 0,0001 em unidades de variância foram mantidos. Os originais estão em `backup_validacao_20260909/`.

## Critérios de comparação

A comparação principal inclui GARCH, HAR-RV e as 12 configurações de ML (três modelos × quatro conjuntos de informações). Para cada praça e horizonte, utiliza apenas as datas com previsão válida em todas essas configurações. O EGARCH recebe uma tabela adicional em suporte comum que também exige previsão válida desse modelo. A cobertura de todos os modelos é informada separadamente.

QLIKE é calculado sobre variância. MAE e RMSE são calculados sobre volatilidade. O RMSE agregado é a raiz da média dos erros quadráticos individuais; não é a média de RMSEs anuais. A tabela agregada pondera cada previsão igualmente, de modo que praças com mais observações têm maior peso. A tabela por praça permite avaliar a heterogeneidade.

Os ganhos incrementais são calculados como `100 × (1 − erro_experimento / erro_histórico_próprio)` para cada modelo, praça e horizonte. Valores positivos indicam redução de erro. São comparações descritivas; não representam significância estatística.

## Arquivos e reprodução

- `temporal_validation.py`: separação anual e registros individuais.
- `04_baselines_garch_har.py`: reexecução dos baselines.
- `05_ml_experiments.py`: reexecução das 12 configurações de ML.
- `07_compare_validated.py`: auditoria, suporte comum, métricas e ganhos.
- `test_temporal_validation.py`: testes de regressão.

Executar na raiz do estudo, em PowerShell:

```powershell
.\.mlf\Scripts\python.exe -m unittest test_temporal_validation -v
.\.mlf\Scripts\python.exe 04_baselines_garch_har.py
.\.mlf\Scripts\python.exe 05_ml_experiments.py
.\.mlf\Scripts\python.exe 07_compare_validated.py
```

As saídas ficam em `dados_commodities_bahia/`: `baseline_predictions_cafe.csv`, `ml_predictions_cafe.csv`, `validated_fold_audit.csv`, `validated_coverage.csv`, `validated_comparison_by_series.csv`, `validated_comparison_pooled.csv`, `validated_comparison_including_egarch.csv`, `validated_incremental_gains.csv` e `validated_run_manifest.json`.

## Limites desta etapa

Esta correção trata a disponibilidade dos alvos no corte de treinamento e o alinhamento da origem dos modelos. Não certifica os horários históricos de publicação das variáveis externas, que foram integradas por data. A utilização de informação externa no mesmo dia pressupõe sua disponibilidade antes da emissão da previsão; atrasos de divulgação exigem uma auditoria própria.

Permanecem pendentes os testes estatísticos de diferenças preditivas, a sensibilidade ao piso QLIKE e às lacunas de cotação, a avaliação por regime e a otimização interna de hiperparâmetros. E2 reúne outras commodities e benchmarks de café; seu ganho não isola o efeito de soja, algodão e cacau.

O arquivo `forecast_current_cafe.csv` não foi atualizado nesta etapa. Ele foi gerado anteriormente com outra configuração de Random Forest e com cotação-base de 21/01/2026; não deve ser tratado como previsão atual validada por esta comparação.
