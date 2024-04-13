#	Engenharia de recursos - Em busca das melhores práticas

## 1.	Introdução

Este texto tem por objetivo apresenta um Elaborar um resumo detalhado do capítulo 5 do Livro "Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications, Chip Huyen". Onde serão enfatizadas as principais estratégias e técnicas de engenharia de recursos (feature engineering) destacadas pela autora. Através de uma análise dos insights e conceitos-chave do capítulo, será oferecido um panorama claro e conciso do conteúdo abordado. As práticas mais críticas para engenharia de recursos serão pormenorizadas, buscando explicar o porquê delas serem tão importantes e como contribuem para o sucesso de um modelo de aprendizado de máquina. Por fim, serão apresentadas as melhores práticas identificadas no capítulo, destacando sua relevância e impacto no desenvolvimento de modelos eficazes de aprendizado de máquina.

## 2.	Engenharia de recursos

Os recursos certos é a coisa mais importante no desenvolvimento de modelos de machine learning (ML). Muitas empresas têm observado que investir em recursos adequados após obter um modelo viável pode levar ao maior aumento de desempenho, superando até mesmo técnicas avançadas de ajuste de hiperparâmetros. A engenharia de recursos se tornou uma parte crucial do trabalho em ML e ciência de dados, focando na criação de novos recursos úteis. 

A importância da engenharia de recursos pode ser questionada, especialmente considerando a promessa do aprendizado profundo de eliminar a necessidade de projetar recursos manualmente. Embora o aprendizado profundo possa automatizar a extração de muitos recursos, ainda não atingimos o ponto em que todos os recursos podem ser automatizados, e a maioria dos aplicativos de ML em produção ainda não são baseados em aprendizado profundo. A engenharia de recursos requer conhecimento específico do domínio e tende a ser um processo iterativo e frágil. No entanto, com o surgimento do aprendizado profundo, grande parte dessa complexidade foi aliviada, tornando a extração de recursos para texto e imagens mais automatizada. Para tarefas complexas, como recomendações de vídeos ou detecção de fraudes, ainda pode ser necessário extrair e utilizar milhões de recursos, exigindo conhecimento especializado no domínio para criar recursos úteis.

A engenharia de recursos é essencial em projetos de ML e possui técnicas para agilizar o processo, como tratamento de valores ausentes, dimensionamento, discretização, codificação de recursos categóricos e geração de recursos cruzados.

### 2.1.	Como lidar com valores ausentes

Ao lidar com dados em produção, é comum encontrar valores ausentes, mas é importante entender que nem todos os tipos de valores ausentes são iguais. Existem três tipos principais: ausente não aleatoriamente (MNAR), quando a falta de um valor está relacionada ao próprio valor (por exemplo, renda não divulgada); ausentes aleatoriamente (MAR), quando a falta de um valor está relacionada a outra variável observada (por exemplo, idade não divulgada por pessoas de um certo gênero); e ausente completamente aleatoriamente (MCAR), quando a falta de um valor não segue nenhum padrão discernível. Ao encontrar valores ausentes, você pode optar por preenchê-los com valores específicos (imputação) ou remover os valores ausentes (exclusão).

### 2.2.	Eliminação

Muitos tendem a preferir a exclusão de valores ausentes, não por ser o melhor método, mas por ser mais simples. A exclusão pode ser feita de duas maneiras: exclusão de coluna, onde uma variável com muitos valores ausentes é removida, e exclusão de linha, onde uma amostra com valores ausentes é removida. Ambos os métodos têm suas desvantagens, como a perda de informações importantes e a introdução de vieses no modelo. Por exemplo, remover a variável "Estado civil" com muitos valores faltantes pode resultar na perda de informações relevantes, como a correlação entre o estado civil e a compra de casas. Além disso, remover linhas com valores ausentes pode introduzir viés no modelo, especialmente se os valores faltantes não forem aleatórios. Portanto, a exclusão deve ser feita com cautela, considerando o impacto na precisão e na qualidade do modelo.

### 2.3.	Imputação

Embora a exclusão seja uma opção fácil, pode resultar na perda de informações cruciais e introduzir vieses no modelo. A imputação, preenchendo os valores ausentes com determinados valores, é uma alternativa. No entanto, decidir quais valores usar pode ser desafiador. Uma prática comum é usar valores padrão ou estatísticas como média, mediana ou moda para preencher os valores ausentes. Essas práticas funcionam bem na maioria dos casos, mas também podem causar problemas. É importante evitar preencher valores ausentes com valores possíveis, pois isso pode distorcer as previsões do modelo. Diversas técnicas podem ser empregadas em conjunto para lidar com valores ausentes, mas não há uma abordagem perfeita. Tanto a exclusão quanto a imputação têm suas limitações e devem ser usadas com cautela para evitar preconceitos e ruídos nos dados.

### 2.4.	Dimensionamento

Quando os dados têm escalas diferentes, como idade variando de 20 a 40 anos e renda anual variando de 10.000 a 150.000, é crucial dimensioná-los antes de usá-los em modelos de ML. O dimensionamento de recursos é fundamental para evitar que o modelo dê mais importância a variáveis com valores maiores, independentemente da relevância real para as previsões. Uma abordagem comum é colocar os dados no intervalo [0, 1] ou normalizá-los para ter média zero e variância unitária, especialmente se os dados seguirem uma distribuição normal. No entanto, modelos de ML podem ter dificuldades com dados que possuem distribuições distorcidas. Nesses casos, a transformação logarítmica pode ajudar a reduzir a assimetria dos dados. Embora essa técnica possa melhorar o desempenho em muitos casos, é importante analisar os dados transformados em vez dos originais. O dimensionamento também pode ser uma fonte de vazamento de dados se as estatísticas utilizadas para dimensionar os dados não forem atualizadas para refletir mudanças nos dados de entrada durante a inferência. Portanto, é recomendável reavaliar o modelo com frequência para evitar problemas decorrentes de mudanças nos dados.

### 2.5.	Discretização

A discretização é uma técnica que transforma recursos contínuos em categorias discretas, facilitando o treinamento de modelos, especialmente com dados limitados. Por exemplo, a renda anual pode ser agrupada em categorias como baixa, média e alta, simplificando o aprendizado do modelo. No entanto, a escolha dos limites das categorias pode ser difícil e introduzir descontinuidades nos dados. É importante usar o bom senso, quantis básicos e conhecimento do assunto para escolher os limites das categorias de forma mais eficaz.

### 2.6.	Codificando recursos categóricos

Lidar com características categóricas que podem mudar ao longo do tempo, ao contrário de categorias estáticas como faixas etárias. Por exemplo, ao tentar prever compras na Amazon com base na marca do produto, enfrentamos o desafio de lidar com milhões de marcas em constante evolução. Uma abordagem comum é codificar categorias menos comuns como "DESCONHECIDO" para evitar que o modelo trave ao encontrar uma marca nova durante a produção. No entanto, isso pode levar a problemas, como não recomendar produtos de novas marcas simplesmente porque não foram vistas durante o treinamento.

Para resolver esse problema, uma solução é utilizar um truque de hash, como o Vowpal Wabbit. Esse método converte categorias em índices fixos de maneira eficiente, permitindo que o modelo lide com novas categorias sem a necessidade de retreiná-lo. Isso é especialmente útil em ambientes onde novas categorias ou marcas são frequentemente introduzidas. Embora alguns acadêmicos possam considerá-lo um "truque", na prática, é amplamente adotado na indústria devido à sua eficácia comprovada. Além disso, esse método é suportado por várias bibliotecas populares de aprendizado de máquina, como scikit-learn, TensorFlow e gensim, tornando-o acessível e fácil de implementar em diferentes cenários.

### 2.7.	Cruzamento de recursos

O cruzamento de recursos é uma técnica para combinar dois ou mais recursos e gerar novos. Essa técnica é útil para modelar relações não lineares entre variáveis. É especialmente importante para modelos como regressão linear e logística, além de modelos baseados em árvores, que têm dificuldade em aprender essas relações não lineares. Embora menos crucial para redes neurais, ainda pode ser útil. No entanto, o cruzamento de recursos pode aumentar o espaço de recursos, exigindo mais dados para o modelo aprender, podendo também levar ao overfit, especialmente ao usar muitos recursos.

### 2.8.	Incorporações posicionais discretas e contínuas

A incorporação posicional, introduzida no artigo "Atenção é tudo que você precisa", é uma técnica fundamental em visão computacional e processamento de linguagem natural. Ela é necessária para modelos como o Transformer, que processam palavras em paralelo e exigem a ordem explícita das palavras para entender o contexto. Para evitar inserir posições absolutas, que podem prejudicar o desempenho do modelo, as posições são redimensionadas para um intervalo entre 0 e 1. A incorporação posicional é tratada da mesma forma que a incorporação de palavras, usando uma matriz de incorporação com o número de colunas igual ao número de posições. As incorporações de posição podem ser aprendidas, como no BERT, ou fixos, gerados com funções seno e cosseno, chamados de recursos de Fourier, que melhoram o desempenho do modelo para tarefas que utilizam coordenadas ou posições.

## 3.	Vazamento de informações

A MIT Technology Review publicou um artigo em julho de 2021 sobre como centenas de ferramentas de IA desenvolvidas para prever os riscos da COVID-19 em exames médicos não foram úteis na prática. Alguns modelos acabaram aprendendo a prever com base na posição dos pacientes durante os exames, enquanto outros captavam informações específicas de hospitais que os rotulavam, levando a previsões incorretas. Isso é conhecido como vazamento de dados, quando informações do rótulo são "vazadas" para o conjunto de recursos usados para fazer previsões, mas não estão disponíveis durante a inferência, ou seja, informações externas não intencionais são incorporadas aos dados de treinamento de um modelo, influenciando indevidamente seu desempenho. O vazamento de dados pode levar a falhas inesperadas nos modelos, mesmo após testes extensivos.
Um exemplo adicional mostra como um modelo treinado para prever sinais de câncer em tomografias computadorizadas teve desempenho excelente em um hospital, mas ruim em outro. Isso ocorreu porque o um hospital utilizava uma máquina de tomografia mais avançada para casos suspeitos, o que afetou as imagens e foi captado pelo modelo. Por outro lado, um outro hospital utilizava máquinas diferentes de forma aleatória, resultando em um modelo menos confiável. O vazamento de dados é um desafio comum e muitas vezes não é abordado de forma adequada pelos cientistas de dados.

### 3.1.	Causas comuns para vazamento de dados
	
#### 3.1.1.	Dividindo dados correlacionados com o tempo aleatoriamente em vez de por tempo

Uma das causas mais comuns de vazamento de dados é a divisão aleatória de dados correlacionados com o tempo em conjuntos de treinamento, validação e teste. A correlação temporal pode afetar a distribuição dos rótulos, mesmo quando não é óbvia. Por exemplo, ao prever cliques em recomendações musicais, a tendência musical de um dia pode influenciar os resultados. Dividir os dados aleatoriamente pode vazar informações futuras para o modelo, levando a previsões imprecisas. Para evitar isso, é recomendável dividir os dados por tempo sempre que possível, usando os dados mais antigos para treinamento e os mais recentes para validação e teste.

#### 3.1.2.	Dimensionar antes de dividir
É importante dimensionar os recursos dos dados, o que requer estatísticas globais, como média e variância. Um erro comum é calcular essas estatísticas usando todos os dados antes de dividi-los em conjuntos de treinamento, validação e teste, o que pode vazar informações do conjunto de teste para o treinamento. Para evitar esse vazamento, é recomendável dividir os dados antes de dimensioná-los e usar as estatísticas apenas do conjunto de treinamento para o dimensionamento de todas as divisões.

#### 3.1.3.	Preenchendo dados ausentes com estatísticas da divisão de teste

Para lidar com valores ausentes em um recurso é comum preenchê-los com a média ou mediana dos valores existentes. No entanto, calcular a média ou mediana usando todos os dados, em vez de apenas os dados de treinamento, pode causar vazamento de informações do conjunto de teste para o treinamento. Para evitar esse vazamento, é recomendável calcular a média ou mediana apenas com os dados de treinamento e usar essas estatísticas para preencher os valores ausentes em todas as divisões.

#### 3.1.4.	Tratamento inadequado da duplicação de dados antes da divisão

Certifique-se de verificar e remover duplicatas nos seus dados antes e depois de dividir, para evitar que as mesmas amostras apareçam em diferentes divisões. Se você precisar realizar sobreamostragem, faça isso somente após a divisão dos dados. Essas medidas são essenciais para garantir que seu modelo não seja treinado em dados duplicados, o que poderia distorcer suas previsões e levar a resultados incorretos.

#### 3.1.5.	Vazamento de grupo

Um tipo de vazamento de dados que ocorre quando exemplos com rótulos correlacionados são divididos em diferentes conjuntos. Por exemplo, um paciente faz duas tomografias computadorizadas de pulmão com uma semana de diferença, ambas com rótulos sobre sinais de câncer, mas uma é usada no treinamento e a outra no teste. Isso também pode acontecer em detecção de objetos, onde fotos do mesmo objeto tiradas rapidamente são divididas entre treino e teste. Evitar esse vazamento é difícil sem entender a geração dos dados.

#### 3.1.6.	Vazamento do processo de geração de dados

Esse tipo de vazamento ocorre quando informações sobre os dados de teste vazam para o treinamento devido a diferenças nos processos de coleta de dados. Detectar esse vazamento requer um profundo conhecimento da coleta de dados. Embora não haja uma maneira infalível de evitá-lo, você pode mitigar o risco normalizando seus dados e envolvendo especialistas no assunto durante o projeto de ML.

#### 3.1.7.	Detectando vazamento de dados

O vazamento de dados pode ocorrer em várias etapas do processo, desde a geração e coleta até a engenharia de recursos. É crucial monitorar o vazamento durante todo o ciclo de vida do projeto de ML. Medir o poder preditivo de cada recurso em relação à variável alvo e fazer estudos de ablação para entender a importância de cada recurso são práticas recomendadas. Além disso, é importante ter cuidado ao adicionar novos recursos e ao usar a divisão de teste para evitar vazamentos de dados.

## 4.	Bons recursos de engenharia

Adicionar mais recursos geralmente melhora o desempenho do modelo, mas ter muitos pode ser prejudicial. Muitos recursos aumentam o risco de vazamento de dados, overfitting e exigem mais recursos computacionais. Recursos inúteis podem se tornar dívidas técnicas. É importante avaliar a importância e a generalização dos recursos para decidir quais manter. Armazenar recursos removidos e definições gerais de recursos pode ser útil para reutilização e compartilhamento.

## 5.	Importância do recurso

Existem diferentes métodos para medir a importância de recursos em um modelo de ML. O XGBoost oferece funções integradas para isso, enquanto o SHAP (SHapley Additive exPlanations) é útil para métodos mais independentes de modelo. O SHAP não só mede a importância de um recurso para o modelo como um todo, mas também a contribuição de cada recurso para previsões específicas. Identificar os principais recursos pode ser crucial, já que um pequeno número deles muitas vezes é responsável por grande parte da importância do modelo. Essas técnicas não apenas ajudam a escolher os melhores recursos, mas também melhoram a interpretabilidade do modelo.

## 6.	Generalização de recursos

O objetivo de um modelo de ML é fazer previsões precisas em dados não vistos, exigindo que os recursos sejam generalizáveis. Nem todos os recursos têm a mesma capacidade de generalização. Por exemplo, em um modelo para prever se um comentário é spam, o identificador único de cada comentário não é generalizável, mas o identificador do usuário pode ser útil.

Medir a generalização dos recursos requer intuição e experiência. A cobertura de um recurso é a porcentagem de amostras que possuem valores para esse recurso nos dados, sendo baixa cobertura um sinal de falta de generalização. Porém, mesmo recursos com baixa cobertura podem ser úteis se a falta de valores não for aleatória. Além disso, a distribuição dos valores de um recurso entre dados vistos e não vistos deve ser considerada para garantir a generalização adequada.

A generalização de um recurso requer uma compensação entre generalidade e especificidade. Por exemplo, um recurso que indica se é horário de pico pode ser mais generalizável, mas menos específico do que apenas o horário do dia. Essas considerações são essenciais para garantir que o modelo generalize bem para novos dados.

## 7.	Conclusões
A engenharia de recursos desempenha um papel crucial no desenvolvimento de modelos de machine learning, sendo responsável por criar novos recursos que aprimoram o desempenho dos modelos. Embora o aprendizado profundo tenha avançado na automação da extração de recursos, ainda é comum a necessidade de projetar recursos manualmente em muitos casos. Operações como tratamento de valores ausentes, dimensionamento, discretização, codificação de recursos categóricos e geração de recursos cruzados são essenciais nesse processo.

O tratamento de valores ausentes, por exemplo, requer uma escolha cuidadosa entre exclusão e imputação, uma vez que cada abordagem possui suas vantagens e desvantagens. O dimensionamento de recursos e a discretização são cruciais para garantir que o modelo não dê mais peso a variáveis com valores maiores, enquanto a codificação de recursos categóricos e o cruzamento de recursos são úteis para lidar com dados complexos e não lineares.

Evitar o vazamento de dados é fundamental para garantir a integridade do modelo, pois o vazamento pode levar a previsões incorretas. Para isso, é importante seguir boas práticas, como dividir os dados por tempo, dimensionar e normalizar os dados após a divisão e usar estatísticas apenas da divisão de treinamento.

Por fim, avaliar a importância e a generalização dos recursos é essencial para manter apenas os recursos mais relevantes para o modelo. Métodos como XGBoost e SHAP são úteis para medir a importância dos recursos e melhorar a interpretabilidade do modelo. Em resumo, a engenharia de recursos desempenha um papel crítico no desenvolvimento de modelos de ML de alta qualidade e deve ser cuidadosamente considerada em todas as etapas do processo de modelagem.

Finalmente, após todas as considerações e análises apresentadas, podemos definir as melhores práticas da engenharia de recursos.

## 8.	Sintese das melhores práticas da engenharia de recursos

-	Divida os dados por tempo em divisões de treinamento/válido/teste, em vez de fazê-lo aleatoriamente.
-	Se você fizer uma sobreamostragem dos dados, faça-o após a divisão.
-	Dimensione e normalize seus dados após a divisão para evitar vazamento de dados.
-	Use estatísticas apenas da divisão do trem, em vez de todos os dados, para dimensionar seus recursos e lidar com valores ausentes.
-	Entenda como seus dados são gerados, coletados e processados. Envolva especialistas no domínio, se possível.
-	Acompanhe a linhagem dos seus dados.
-	Entenda a importância dos recursos para o seu modelo.
-	Use recursos que generalizem bem.
-	Remova recursos que não são mais úteis de seus modelos.
