#%%
import streamlit as st

st.title("Sobre o Projeto")

st.markdown("Essa página busca detalhar um pouco sobre o processo de construção do modelo.")
st.markdown("---")
st.markdown('## Modelo: Random Forest')
st.markdown('O núcleo do projeto de classificação é um modelo de **Random Forest**. Trata-se de um poderoso algoritmo de aprendizagem automática do tipo *Ensemble*.')
st.markdown('Em vez de depender de uma única árvore de decisão, o Random Forest cria centenas de Árvores de Decisão independentes durante a fase de treino. Cada árvore avalia os dados do cliente de forma ligeiramente diferente. Quando um novo pedido de empréstimo entra no sistema, todas as árvores "votam" no resultado. Então é feita uma média desses "votos" para que seja eleita uma previsão final.')

st.markdown('**Vantagens desta escolha:**')
st.markdown('* Elevada precisão na classificação de dados complexos.')
st.markdown('* Excelente resistência ao *overfitting* (memorização excessiva dos dados de treino).')
st.markdown('* Capacidade nativa de lidar com relações não lineares entre as características do cliente e o risco de crédito.')
st.markdown('Para otimização de quais seriam os melhores hiperparâmetros (`n_estimators`, `min_samples_leaf`, `criterion`), foi utilizado o **GridSearchCV** para realizar uma pesquisa exaustiva e automatizada pelas melhores afinações.')
st.markdown('Após a otimização do **GridSearchCV** foi determinado que o melhor critério matemático para a construção das árvores do Random Forest foi a **entropia de Shannon**. Mas como exatamente isso ajuda o modelo a aprovar ou rejeitar empréstimos?')
st.markdown('A entropia de Shannon é um conceito da Teoria da Informação criada por **Claude Shannon** em 1948 com a publicação de seu artigo *"A Mathematical Theory of Communication"*. É uma métrica que mede o nivel de impureza num grupo de clientes. O objetivo do modelo é aplicar regras de corte para separar nesse caso os bons pagadores dos maus pagadores, até formar grupos mais puros possível para uma melhor decisão.')
st.markdown('A cada passo do algoritmo, é testado cortes em todas as variáveis (renda, valor do empréstimo) e escolhe a pergunta que mais reduz a Entropia do grupo, o que é chamado de *Ganho de informação*. A entropia de Shannon é calculada pela seguinte fórmula:')
st.latex('H(S) = - \sum_{i=1}^{c} p_i \log_2 (p_i)')
st.markdown("#### Exemplo")
st.markdown(
    "Imagine um grupo inicial $S$ contendo **100 clientes**: "
    "**60 bons pagadores** ($p_0 = 0.6$) e **40 em risco** ($p_1 = 0.4$)."
)

st.markdown("1. **Cálculo da Entropia Inicial $H(S)$:**")
st.latex(r"H(S) = - \left( 0.6 \cdot \log_2(0.6) + 0.4 \cdot \log_2(0.4) \right)")
st.latex(r"H(S) = - \left( 0.6 \cdot (-0.737) + 0.4 \cdot (-1.322) \right)")
st.latex(r"H(S) = - (-0.442 - 0.529) = 0.971")

st.markdown(
    "Como o resultado está próximo de $1.0$, o grupo inicial possui **alta incerteza/impureza**."
)

st.markdown("2. **Testando uma Regra de Corte (Ex: *Rendimento > 80.000*):**")
st.markdown(
    "Suponha que a regra dividiu o grupo em dois subgrupos de 50 pessoas:\n"
    "* **Subgrupo 1 ($S_1$):** 45 bons pagadores e 5 em risco ($p_0 = 0.9, p_1 = 0.1$).\n"
    "* **Subgrupo 2 ($S_2$):** 15 bons pagadores e 35 em risco ($p_0 = 0.3, p_1 = 0.7$)."
)

st.markdown("Calculando a Entropia dos novos subgrupos:")
st.latex(r"H(S_1) = - \left( 0.9 \log_2(0.9) + 0.1 \log_2(0.1) \right) \approx 0.469")
st.latex(r"H(S_2) = - \left( 0.3 \log_2(0.3) + 0.7 \log_2(0.7) \right) \approx 0.882")

st.markdown("3. **Cálculo do Ganho de Informação ($IG$):**")
st.markdown(
    "O **Ganho de Informação** mede a redução da entropia obtida após a divisão pelo atributo $A$:"
)
st.latex(r"IG(S, A) = H(S) - \sum_{v \in Valores(A)} \frac{|S_v|}{|S|} H(S_v)")

st.markdown("Aplicando os valores ponderados dos dois subgrupos:")
st.latex(r"IG(S, A) = 0.971 - \left( 0.5 \cdot 0.469 + 0.5 \cdot 0.882 \right)")
st.latex(r"IG(S, A) = 0.971 - 0.6755 = 0.2955")
st.markdown('O ganho de informação desta regra como podemos ver foi $0.2955$. O Random Forest vai repetir esse mesmo cálculo para as outras variáveis (features: histórico de credito, valor do empréstimo etc). No final, ele escolhe aplicar a regra que gere o maior ganho de informação, garantindo que as ramificações das árvores tornem os dados o mais puros possíveis.')
st.markdown('---')

st.markdown('## Análise Exploratória')
st.markdown(
    "Importante ressaltar que inicialmente eu separei o dataset em Treino e Teste (80% treino e 20% teste), e toda a análise exploratória foi feita na base de treino. Para a análise em si, escolhi aplicar uma abordagem "
    "orientada a dados para identificar quais as variáveis que realmente impactavam a aprovação "
    "ou o risco de crédito, eliminando ruído e reduzindo a dimensionalidade do problema."
)
st.markdown(
    "Treinei uma **Árvore de Decisão** (`DecisionTreeClassifier`) sobre os dados "
    "historicos para calcular o ganho de pureza trazido por cada variável através da métrica de "
    "**Importância de Gini/Entropia** (`feature_importances_`)."
)
st.markdown(
    "Depois realizei um filtro de importância cumulativa (Corte a 96%). Em vez de definir um número arbitrário de colunas (ex: 'manter as 5 melhores'), utilizei "
    "a **Soma Cumulativa do Poder Explicativo**:"
)
st.markdown(
    "* Todas as variáveis foram ordenadas da mais importante para a menos importante.\n"
    "* Calculou-se a importância acumulada percentual de cada variável na decisão.\n"
    "* **Ponto de Corte:** Retiveram-se apenas as variáveis necessárias para atingir **96% da informação acumulada** (`acum. < 0.96`), descartando as restantes que adicionavam pouca ou nenhuma relevância."
)
st.markdown("#### Vantagens desta Abordagem")
st.markdown("* **Redução de Ruído:** Remove variáveis irrelevantes ou redundantes antes do treino final.")
st.markdown("* **Eficiência Computacional:** Acelera o `GridSearchCV` sem perder capacidade preditiva.")
st.markdown("* **Generalização:** Evita que o modelo crie regras baseadas em colunas com baixo poder explicativo")
st.markdown('---')

st.markdown('## Métricas')
st.markdown(
    "Para validar a eficácia do modelo, avaliei "
    "o desempenho no conjunto de dados de teste (dados nunca antes vistos pelo modelo durante o treino). "
    "Abaixo estão detalhadas as principais métricas e ferramentas de avaliação utilizadas."
)
st.markdown("#### ACC")
st.markdown('ACC (ou acurácia) mede a proporção geral de acertos sobre o total dos casos analisados.')
st.latex(r"\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}")
st.markdown('A acurácia obtida na base de testes foi de **91%**')

st.markdown("#### Curva ROC e Área sob a Curva (AUC)")
st.markdown(
    "A curva **ROC (Receiver Operating Characteristic)** avalia a taxa de Verdadeiros Positivos contra "
    "a taxa de Falsos Positivos em múltiplos limiares de decisão (*thresholds*)."
)
st.image(
    "app/img/output.png",
    caption="ROC Curve",
)
st.markdown(
    "A **AUC (Area Under Curve)** quantifica essa capacidade de separação global em um único número de 0 a 1:\n"
    "* **AUC = 0.50:** O modelo decide de forma aleatória (equivalente a atirar uma moeda ao ar).\n"
    "* **AUC > 0.80:** Excelente capacidade discriminatória.\n"
    "* **AUC > 0.90:** Desempenho excecional na separação dos perfis de crédito."
)
st.markdown("O modelo teve uma AUC na base de test de **96,6%**.")
st.markdown('---')

st.markdown("## MLOps: Ciclo de Vida do Modelo")
st.markdown(
    "Para além da construção do modelo de Machine Learning, o projeto foi estruturado segundo "
    "os princípios de **MLOps (Machine Learning Operations)**. Isto garante que todo o pipeline "
    "seja rastreável, reprodutível, versionado e pronto para consumo em ambiente de produção."
)

st.markdown("#### Rastreamento de Experimentos e Versionamento (MLflow)")
st.markdown(
    "Utilizei o **MLflow** como plataforma central para gerir o ciclo de vida dos modelos. "
    "Através do registro automático do MLflow, cada execução de treino é "
    "armazenada, podendo comparar as diferentes runs, para acompanhar de maneira detalhada quais mudanças estão impactando mais na performance do modelo."
)

st.markdown(
    "* **Hiperparâmetros:** Registro de todas as combinações testadas no `GridSearchCV` (`n_estimators`, `criterion`, `min_samples_leaf`).\n"
    "* **Métricas de Performance:** Acurácia, ROC-AUC e tempo de execução em cada dobra de validação cruzada.\n"
    "* **Artefactos:** Armazenamento automático da Curva ROC, Matriz de Confusão.\n"
    "* **Model Registry:** Versionamento formal do modelo (ex: `loan_predict_model/v1`, `v2`), permitindo promover apenas as melhores versões para produção e sempre manter o modelo mais atual."
)


st.markdown("#### Pipelines Encapsulados e Reprodutibilidade")
st.markdown(
    "Um dos maiores desafios em MLOps é o *Data Leakage* e a discrepância entre treino e inferência (*Training-Serving Skew*). "
    "Para resolver isto:"
)

st.markdown(
    "* O pré-processamento de dados (remoção de variáveis irrelevantes com `DropFeatures`, "
    "codificação de variáveis categóricas com `OrdinalEncoder` e discretização com `DecisionTreeDiscretiser`) "
    "foi **totalmente encapsulado dentro de um `Pipeline` do Scikit-Learn e Feature-Engine**.\n"
    "* O modelo final guardado não é apenas o classificador, mas o **objeto de pipeline completo**.\n"
    "* Isto garante que a aplicação (seja em Streamlit, ou quaisquer outras aplicações web) possam enviar os dados brutos e o pipeline aplique exatamente as "
    "mesmas transformações aprendidas no treino."
)

st.markdown("## Teste o modelo")

if st.button("Ir para o Modelo", type="primary", use_container_width=True):
    st.switch_page("predict.py")