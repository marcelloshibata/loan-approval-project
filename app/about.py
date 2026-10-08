import streamlit as st

if "lang" not in st.session_state:
    st.session_state.lang = "EN"

col_title, col_lang = st.columns([0.75, 0.25])

with col_lang:
    btn_label = "🇺🇸 English" if st.session_state.lang == "PT" else "🇧🇷 Português"
    if st.button(btn_label, use_container_width=True, type='primary'):
        st.session_state.lang = "EN" if st.session_state.lang == "PT" else "PT"
        st.rerun()

content = {
    "PT": {
        "page_title": "Sobre o Projeto",
        "page_subtitle": "Essa página busca detalhar um pouco sobre o processo de construção do modelo.",
        "rf_title": "## Modelo: Random Forest",
        "rf_p1": "O núcleo do projeto de classificação é um modelo de **Random Forest**. Trata-se de um poderoso algoritmo de aprendizagem automática do tipo *Ensemble*.",
        "rf_p2": 'Em vez de depender de uma única árvore de decisão, o Random Forest cria centenas de Árvores de Decisão independentes durante a fase de treino. Cada árvore avalia os dados do cliente de forma ligeiramente diferente. Quando um novo pedido de empréstimo entra no sistema, todas as árvores "votam" no resultado. Então é feita uma média desses "votos" para que seja eleita uma previsão final.',
        "rf_advantages_title": "**Vantagens desta escolha:**",
        "rf_adv_1": "* Elevada precisão na classificação de dados complexos.",
        "rf_adv_2": "* Excelente resistência ao *overfitting* (memorização excessiva dos dados de treino).",
        "rf_adv_3": "* Capacidade nativa de lidar com relações não lineares entre as características do cliente e o risco de crédito.",
        "gridsearch_p1": "Para otimização de quais seriam os melhores hiperparâmetros (`n_estimators`, `min_samples_leaf`, `criterion`), foi utilizado o **GridSearchCV** para realizar uma pesquisa exaustiva e automatizada pelas melhores afinações.",
        "gridsearch_p2": "Após a otimização do **GridSearchCV** foi determinado que o melhor critério matemático para a construção das árvores do Random Forest foi a **entropia de Shannon**. Mas como exatamente isso ajuda o modelo a aprovar ou rejeitar empréstimos?",
        "entropy_p1": 'A entropia de Shannon é um conceito da Teoria da Informação criada por **Claude Shannon** em 1948 com a publicação de seu artigo *"A Mathematical Theory of Communication"*. É uma métrica que mede o nivel de impureza num grupo de clientes. O objetivo do modelo é aplicar regras de corte para separar nesse caso os bons pagadores dos maus pagadores, até formar grupos mais puros possível para uma melhor decisão.',
        "entropy_p2": "A cada passo do algoritmo, é testado cortes em todas as variáveis (renda, valor do empréstimo) e escolhe a pergunta que mais reduz a Entropia do grupo, o que é chamado de *Ganho de informação*. A entropia de Shannon é calculada pela seguinte fórmula:",
        "example_title": "#### Exemplo",
        "example_intro": "Imagine um grupo inicial $S$ contendo **100 clientes**: **60 bons pagadores** ($p_0 = 0.6$) e **40 em risco** ($p_1 = 0.4$).",
        "step1_title": "1. **Cálculo da Entropia Inicial $H(S)$:**",
        "step1_interp": "Como o resultado está próximo de $1.0$, o grupo inicial possui **alta incerteza/impureza**.",
        "step2_title": "2. **Testando uma Regra de Corte (Ex: *Rendimento > 80.000*):**",
        "step2_body": "Suponha que a regra dividiu o grupo em dois subgrupos de 50 pessoas:\n* **Subgrupo 1 ($S_1$):** 45 bons pagadores e 5 em risco ($p_0 = 0.9, p_1 = 0.1$).\n* **Subgrupo 2 ($S_2$):** 15 bons pagadores e 35 em risco ($p_0 = 0.3, p_1 = 0.7$).",
        "step2_calc": "Calculando a Entropia dos novos subgrupos:",
        "step3_title": "3. **Cálculo do Ganho de Informação ($IG$):**",
        "step3_desc": "O **Ganho de Informação** mede a redução da entropia obtida após a divisão pelo atributo $A$:",
        "step3_apply": "Aplicando os valores ponderados dos dois subgrupos:",
        "example_conclusion": "O ganho de informação desta regra como podemos ver foi $0.2955$. O Random Forest vai repetir esse mesmo cálculo para as outras variáveis (features: histórico de credito, valor do empréstimo etc). No final, ele escolhe aplicar a regra que gere o maior ganho de informação, garantindo que as ramificações das árvores tornem os dados o mais puros possíveis.",
        "eda_title": "## Análise Exploratória",
        "eda_p1": "Importante ressaltar que inicialmente eu separei o dataset em Treino e Teste (80% treino e 20% teste), e toda a análise exploratória foi feita na base de treino. Para a análise em si, escolhi aplicar uma abordagem orientada a dados para identificar quais as variáveis que realmente impactavam a aprovação ou o risco de crédito, eliminando ruído e reduzindo a dimensionalidade do problema.",
        "eda_p2": "Treinei uma **Árvore de Decisão** (`DecisionTreeClassifier`) sobre os dados historicos para calcular o ganho de pureza trazido por cada variável através da métrica de **Importância de Gini/Entropia** (`feature_importances_`).",
        "eda_p3": "Depois realizei um filtro de importância cumulativa (Corte a 96%). Em vez de definir um número arbitrário de colunas (ex: 'manter as 5 melhores'), utilizei a **Soma Cumulativa do Poder Explicativo**:",
        "eda_bullets": "* Todas as variáveis foram ordenadas da mais importante para a menos importante.\n* Calculou-se a importância acumulada percentual de cada variável na decisão.\n* **Ponto de Corte:** Retiveram-se apenas as variáveis necessárias para atingir **96% da informação acumulada** (`acum. < 0.96`), descartando as restantes que adicionavam pouca ou nenhuma relevância.",
        "eda_advantages_title": "#### Vantagens desta Abordagem",
        "eda_adv_1": "* **Redução de Ruído:** Remove variáveis irrelevantes ou redundantes antes do treino final.",
        "eda_adv_2": "* **Eficiência Computacional:** Acelera o `GridSearchCV` sem perder capacidade preditiva.",
        "eda_adv_3": "* **Generalização:** Evita que o modelo crie regras baseadas em colunas com baixo poder explicativo",
        "metrics_title": "## Métricas",
        "metrics_intro": "Para validar a eficácia do modelo, avaliei o desempenho no conjunto de dados de teste (dados nunca antes vistos pelo modelo durante o treino). Abaixo estão detalhadas as principais métricas e ferramentas de avaliação utilizadas.",
        "acc_title": "#### ACC",
        "acc_desc": "ACC (ou acurácia) mede a proporção geral de acertos sobre o total dos casos analisados.",
        "acc_result": "A acurácia obtida na base de testes foi de **91%**",
        "roc_title": "#### Curva ROC e Área sob a Curva (AUC)",
        "roc_desc": "A curva **ROC (Receiver Operating Characteristic)** avalia a taxa de Verdadeiros Positivos contra a taxa de Falsos Positivos em múltiplos limiares de decisão (*thresholds*).",
        "auc_desc": "A **AUC (Area Under Curve)** quantifica essa capacidade de separação global em um único número de 0 a 1:\n* **AUC = 0.50:** O modelo decide de forma aleatória (equivalente a atirar uma moeda ao ar).\n* **AUC > 0.80:** Excelente capacidade discriminatória.\n* **AUC > 0.90:** Desempenho excecional na separação dos perfis de crédito.",
        "auc_result": "O modelo teve uma AUC na base de test de **96,6%**.",
        "mlops_title": "## MLOps: Ciclo de Vida do Modelo",
        "mlops_intro": "Para além da construção do modelo de Machine Learning, o projeto foi estruturado segundo os princípios de **MLOps (Machine Learning Operations)**. Isto garante que todo o pipeline seja rastreável, reprodutível, versionado e pronto para consumo em ambiente de produção.",
        "mlflow_title": "#### Rastreamento de Experimentos e Versionamento (MLflow)",
        "mlflow_desc": "Utilizei o **MLflow** como plataforma central para gerir o ciclo de vida dos modelos. Através do registro automático do MLflow, cada execução de treino é armazenada, podendo comparar as diferentes runs, para acompanhar de maneira detalhada quais mudanças estão impactando mais na performance do modelo.",
        "mlflow_bullets": "* **Hiperparâmetros:** Registro de todas as combinações testadas no `GridSearchCV` (`n_estimators`, `criterion`, `min_samples_leaf`).\n* **Métricas de Performance:** Acurácia, ROC-AUC e tempo de execução em cada dobra de validação cruzada.\n* **Artefactos:** Armazenamento automático da Curva ROC, Matriz de Confusão.\n* **Model Registry:** Versionamento formal do modelo (ex: `loan_predict_model/v1`, `v2`), permitindo promover apenas as melhores versões para produção e sempre manter o modelo mais atual.",
        "pipeline_title": "#### Pipelines Encapsulados e Reprodutibilidade",
        "pipeline_intro": "Um dos maiores desafios em MLOps é o *Data Leakage* e a discrepância entre treino e inferência (*Training-Serving Skew*). Para resolver isto:",
        "pipeline_bullets": "* O pré-processamento de dados (remoção de variáveis irrelevantes com `DropFeatures`, codificação de variáveis categóricas com `OrdinalEncoder` e discretização com `DecisionTreeDiscretiser`) foi **totalmente encapsulado dentro de um `Pipeline` do Scikit-Learn e Feature-Engine**.\n* O modelo final guardado não é apenas o classificador, mas o **objeto de pipeline completo**.\n* Isto garante que a aplicação (seja em Streamlit, ou quaisquer outras aplicações web) possam enviar os dados brutos e o pipeline aplique exatamente as mesmas transformações aprendidas no treino.",
        "cta_title": "## Teste o modelo",
        "cta_button": "Ir para o Modelo",
    },
    "EN": {
        "page_title": "About the Project",
        "page_subtitle": "This page details the process behind building the model.",
        "rf_title": "## Model: Random Forest",
        "rf_p1": "The core of the classification project is a **Random Forest** model. It is a powerful *Ensemble* machine learning algorithm.",
        "rf_p2": 'Instead of relying on a single decision tree, Random Forest creates hundreds of independent Decision Trees during training. Each tree evaluates customer data slightly differently. When a new loan request enters the system, all trees "vote" on the outcome. An average of these "votes" is then taken to yield the final prediction.',
        "rf_advantages_title": "**Advantages of this choice:**",
        "rf_adv_1": "* High accuracy in classifying complex data.",
        "rf_adv_2": "* Excellent resistance to *overfitting* (excessive memorization of training data).",
        "rf_adv_3": "* Native ability to handle non-linear relationships between customer attributes and credit risk.",
        "gridsearch_p1": "To optimize the best hyperparameters (`n_estimators`, `min_samples_leaf`, `criterion`), **GridSearchCV** was used to perform an exhaustive and automated search for the best tuning parameters.",
        "gridsearch_p2": "After **GridSearchCV** optimization, **Shannon's Entropy** was determined as the best mathematical criterion for building the Random Forest trees. But how exactly does this help the model approve or reject loans?",
        "entropy_p1": 'Shannon\'s entropy is an Information Theory concept created by **Claude Shannon** in 1948 with the publication of his paper *"A Mathematical Theory of Communication"*. It is a metric that measures the level of impurity in a customer group. The model\'s goal is to apply split rules to separate good payers from high-risk applicants, forming groups that are as pure as possible for better decision-making.',
        "entropy_p2": "At each algorithm step, splits across all variables (income, loan amount) are tested, selecting the question that reduces group Entropy the most, known as *Information Gain*. Shannon's entropy is calculated using the following formula:",
        "example_title": "#### Example",
        "example_intro": "Imagine an initial group $S$ containing **100 customers**: **60 good payers** ($p_0 = 0.6$) and **40 high-risk applicants** ($p_1 = 0.4$).",
        "step1_title": "1. **Calculation of Initial Entropy $H(S)$:**",
        "step1_interp": "Since the result is close to $1.0$, the initial group has **high uncertainty/impurity**.",
        "step2_title": "2. **Testing a Split Rule (Ex: *Income > 80,000*):**",
        "step2_body": "Suppose the rule split the group into two subgroups of 50 people:\n* **Subgroup 1 ($S_1$):** 45 good payers and 5 high-risk ($p_0 = 0.9, p_1 = 0.1$).\n* **Subgroup 2 ($S_2$):** 15 good payers and 35 high-risk ($p_0 = 0.3, p_1 = 0.7$).",
        "step2_calc": "Calculating the Entropy of the new subgroups:",
        "step3_title": "3. **Calculation of Information Gain ($IG$):**",
        "step3_desc": "The **Information Gain** measures the entropy reduction achieved after splitting by attribute $A$:",
        "step3_apply": "Applying the weighted values of both subgroups:",
        "example_conclusion": "As shown, the information gain for this rule was $0.2955$. Random Forest repeats this exact calculation across all other variables (features: credit history, loan amount, etc.). In the end, it selects the split rule that yields the highest information gain, ensuring that tree branches make the data as pure as possible.",
        "eda_title": "## Exploratory Data Analysis",
        "eda_p1": "It is important to note that I initially split the dataset into Train and Test sets (80% train and 20% test), and all exploratory data analysis was conducted on the training set. For the analysis itself, I applied a data-driven approach to identify which variables genuinely impacted loan approval or credit risk, eliminating noise and reducing problem dimensionality.",
        "eda_p2": "I trained a **Decision Tree** (`DecisionTreeClassifier`) on historical data to calculate the purity gain brought by each variable through **Gini/Entropy Importance** (`feature_importances_`).",
        "eda_p3": "Then I applied a cumulative importance filter (Cutoff at 96%). Instead of setting an arbitrary number of columns (e.g., 'keep the top 5'), I used the **Cumulative Sum of Explanatory Power**:",
        "eda_bullets": "* All variables were ranked from most important to least important.\n* The percentage cumulative importance of each variable in the decision was calculated.\n* **Cutoff Point:** Only the variables needed to reach **96% of cumulative information** (`acum. < 0.96`) were retained, discarding the rest that added little to no value.",
        "eda_advantages_title": "#### Advantages of this Approach",
        "eda_adv_1": "* **Noise Reduction:** Removes irrelevant or redundant variables prior to final training.",
        "eda_adv_2": "* **Computational Efficiency:** Speeds up `GridSearchCV` without losing predictive capacity.",
        "eda_adv_3": "* **Generalization:** Prevents the model from learning rules based on low-importance features.",
        "metrics_title": "## Metrics",
        "metrics_intro": "To validate model efficacy, I evaluated performance on the test dataset (data never seen by the model during training). The main evaluation metrics and tools used are detailed below.",
        "acc_title": "#### ACC",
        "acc_desc": "ACC (or accuracy) measures the overall proportion of correct predictions out of all analyzed cases.",
        "acc_result": "The accuracy achieved on the test set was **91%**",
        "roc_title": "#### ROC Curve and Area Under Curve (AUC)",
        "roc_desc": "The **ROC (Receiver Operating Characteristic)** curve evaluates the True Positive Rate against the False Positive Rate across multiple decision thresholds.",
        "auc_desc": "The **AUC (Area Under Curve)** quantifies this global separation capability in a single number from 0 to 1:\n* **AUC = 0.50:** The model makes random decisions (equivalent to flipping a coin).\n* **AUC > 0.80:** Excellent discrimination capacity.\n* **AUC > 0.90:** Exceptional performance in separating credit risk profiles.",
        "auc_result": "The model achieved a test set AUC of **96.6%**.",
        "mlops_title": "## MLOps: Model Lifecycle",
        "mlops_intro": "Beyond model building, the project was structured following **MLOps (Machine Learning Operations)** principles. This ensures the entire pipeline is traceable, reproducible, versioned, and ready for production consumption.",
        "mlflow_title": "#### Experiment Tracking and Versioning (MLflow)",
        "mlflow_desc": "I used **MLflow** as the central platform to manage the model lifecycle. Through MLflow's autologging, each training run is stored, allowing detailed run comparisons to monitor which changes most impact model performance.",
        "mlflow_bullets": "* **Hyperparameters:** Logged all combinations tested in `GridSearchCV` (`n_estimators`, `criterion`, `min_samples_leaf`).\n* **Performance Metrics:** Accuracy, ROC-AUC, and execution time for each cross-validation fold.\n* **Artifacts:** Automatic logging of ROC Curves and Confusion Matrices.\n* **Model Registry:** Formal model versioning (e.g., `loan_predict_model/v1`, `v2`), allowing only the best versions to be promoted to production.",
        "pipeline_title": "#### Encapsulated Pipelines and Reproducibility",
        "pipeline_intro": "A major MLOps challenge is Data Leakage and Training-Serving Skew. To solve this:",
        "pipeline_bullets": "* Data preprocessing (removing irrelevant features with `DropFeatures`, categorical encoding with `OrdinalEncoder`, and discretization with `DecisionTreeDiscretiser`) was **fully encapsulated inside a Scikit-Learn and Feature-Engine `Pipeline`**.\n* The final saved model is not just the classifier, but the **complete pipeline object**.\n* This guarantees that the application (whether in Streamlit or other web applications) can send raw inputs and the pipeline applies the exact same transformations learned during training.",
        "cta_title": "## Test the model",
        "cta_button": "Go to Model",
    },
}

t = content[st.session_state.lang]

st.title(t["page_title"])
st.markdown(t["page_subtitle"])
st.markdown("---")

st.markdown(t["rf_title"])
st.markdown(t["rf_p1"])
st.markdown(t["rf_p2"])

st.markdown(t["rf_advantages_title"])
st.markdown(t["rf_adv_1"])
st.markdown(t["rf_adv_2"])
st.markdown(t["rf_adv_3"])

st.markdown(t["gridsearch_p1"])
st.markdown(t["gridsearch_p2"])

st.markdown(t["entropy_p1"])
st.markdown(t["entropy_p2"])

st.latex(r"H(S) = - \sum_{i=1}^{c} p_i \log_2 (p_i)")

st.markdown(t["example_title"])
st.markdown(t["example_intro"])

st.markdown(t["step1_title"])
st.latex(r"H(S) = - \left( 0.6 \cdot \log_2(0.6) + 0.4 \cdot \log_2(0.4) \right)")
st.latex(r"H(S) = - \left( 0.6 \cdot (-0.737) + 0.4 \cdot (-1.322) \right)")
st.latex(r"H(S) = - (-0.442 - 0.529) = 0.971")
st.markdown(t["step1_interp"])

st.markdown(t["step2_title"])
st.markdown(t["step2_body"])
st.markdown(t["step2_calc"])
st.latex(r"H(S_1) = - \left( 0.9 \log_2(0.9) + 0.1 \log_2(0.1) \right) \approx 0.469")
st.latex(r"H(S_2) = - \left( 0.3 \log_2(0.3) + 0.7 \log_2(0.7) \right) \approx 0.882")

st.markdown(t["step3_title"])
st.markdown(t["step3_desc"])
st.latex(r"IG(S, A) = H(S) - \sum_{v \in Valores(A)} \frac{|S_v|}{|S|} H(S_v)")
st.markdown(t["step3_apply"])
st.latex(r"IG(S, A) = 0.971 - \left( 0.5 \cdot 0.469 + 0.5 \cdot 0.882 \right)")
st.latex(r"IG(S, A) = 0.971 - 0.6755 = 0.2955")

st.markdown(t["example_conclusion"])
st.markdown("---")

st.markdown(t["eda_title"])
st.markdown(t["eda_p1"])
st.markdown(t["eda_p2"])
st.markdown(t["eda_p3"])
st.markdown(t["eda_bullets"])

st.markdown(t["eda_advantages_title"])
st.markdown(t["eda_adv_1"])
st.markdown(t["eda_adv_2"])
st.markdown(t["eda_adv_3"])
st.markdown("---")

st.markdown(t["metrics_title"])
st.markdown(t["metrics_intro"])

st.markdown(t["acc_title"])
st.markdown(t["acc_desc"])
st.latex(r"\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}")
st.markdown(t["acc_result"])

st.markdown(t["roc_title"])
st.markdown(t["roc_desc"])
st.image(
    "app/img/output.png",
    caption="ROC Curve",
)
st.markdown(t["auc_desc"])
st.markdown(t["auc_result"])
st.markdown("---")

st.markdown(t["mlops_title"])
st.markdown(t["mlops_intro"])

st.markdown(t["mlflow_title"])
st.markdown(t["mlflow_desc"])
st.markdown(t["mlflow_bullets"])

st.markdown(t["pipeline_title"])
st.markdown(t["pipeline_intro"])
st.markdown(t["pipeline_bullets"])

st.markdown(t["cta_title"])

if st.button(t["cta_button"], type="primary", use_container_width=True):
    st.switch_page("predict.py")