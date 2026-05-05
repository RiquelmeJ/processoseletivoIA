## 📝 Relatório do Candidato

👤 Identificação:  Riquelme Jatay Ribeiro Scarcela Bezerra


### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura da CNN implementada consiste em um modelo Sequencial simples e enxuto focado em Edge AI. Ela possui:
- 1 camada de entrada para imagens em escala de cinza (28x28x1).
- 1 camada Convolucional (Conv2D) com 32 filtros (kernel 3x3) e ativação ReLU.
- 1 camada de Max Pooling (MaxPooling2D) de tamanho 2x2.
- 1 camada Convolucional (Conv2D) com 64 filtros (kernel 3x3) e ativação ReLU.
- 1 camada de Max Pooling (MaxPooling2D) de tamanho 2x2.
- 1 camada Flatten para vetorização.
- 1 camada Oculta (Dense) com 64 neurônios e ativação ReLU.
- 1 camada de Saída (Dense) com 10 neurônios e ativação Softmax para classificar os dígitos numéricos (0 a 9).


### 2️⃣ Bibliotecas Utilizadas

- TensorFlow e Keras: Utilizados para carregar o dataset (MNIST), construir a rede convolucional, treinar (em CPU) e converter para TFLite.
- OS: Biblioteca nativa do Python, utilizada para lidar com os caminhos dos arquivos de salvamento e forçar o CUDA_VISIBLE_DEVICES como -1 (permitindo apenas o treinamento em CPU).


### 3️⃣ Técnica de Otimização do Modelo

A técnica aplicada foi a Dynamic Range Quantization (Quantização de Faixa Dinâmica) nativa do TensorFlow Lite.
Essa técnica, configurada utilizando tf.lite.Optimize.DEFAULT, converte estaticamente os pesos do modelo treinado de ponto flutuante para inteiros. A partir dela, obtivemos uma redução de mais de 90% no tamanho de armazenamento (de aproximadamente 1.4 MB para cerca de 128 KB), mantendo os desempenhos exigidos para soluções com limites de hardware.


### 4️⃣ Resultados Obtidos

Após 5 épocas de treinamento, a Acurácia Final no conjunto de teste foi de cerca de 98.2%. Além disso, obtivemos uma execução veloz com arquivos de modelo minúsculos, atendendo totalmente às restrições de pipeline de CI e de Edge AI demandadas.


### 5️⃣ Comentários Adicionais 

Durante o desenvolvimento do desafio, um ponto de atenção foi garantir que a complexidade do modelo não prejudicasse o tempo de execução no pipeline de CI e a viabilidade para Edge AI. Isso me guiou à decisão técnica de adotar uma rede convolucional bem enxuta (apenas duas camadas Conv2D intercaladas com Pooling) e forçar a execução inteiramente via CPU, prevenindo gargalos de recursos.

Uma das limitações do modelo é que ele foca primeiramente na velocidade de inferência e no baixo consumo de memória devido à arquitetura intencionalmente minimalista. Isso atende à resolução do MNIST, mas a rede poderia apresentar menor precisão com bases de dados significativamente mais complexas sem um ajuste.

Não enfrentei grandes dificuldades técnicas durante a construção do projeto, mas o principal aprendizado prático foi ter a vivência completa do fluxo de um projeto de EdgeAI. :)

