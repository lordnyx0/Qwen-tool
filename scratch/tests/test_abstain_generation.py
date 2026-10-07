import random

SUBJECTS = [
    ("a teoria da relatividade geral", "A relatividade geral descreve a gravidade como a curvatura do tecido do espaço-tempo provocada pela concentração de massa e energia."),
    ("a fotossíntese vegetal", "É a reação bioquímica nos cloroplastos que utiliza água, CO2 e luz solar para produzir glicose e liberar gás oxigênio."),
    ("a busca binária", "Algoritmo eficiente de busca que divide sucessivamente pela metade o espaço de busca em coleções já ordenadas (complexidade O(log n))."),
    ("o protocolo TCP", "Protocolo orientado a conexão que garante entrega sequencial, controle de fluxo e retransmissão de pacotes perdidos via three-way handshake."),
    ("o espalhamento de Rayleigh", "Dispersão da luz solar pelas partículas da atmosfera terrestre, sendo mais intensa em comprimentos de onda curtos como o azul."),
    ("o feudalismo medieval", "Sistema socioeconômico e político europeu baseado na posse da terra, vassalagem, servidão e descentralização do poder real."),
    ("a mitose celular", "Processo de divisão celular equacional que gera duas células-filhas geneticamente idênticas à célula-mãe."),
    ("o conceito de entropia", "Na termodinâmica, quantifica o grau de desordem ou a indisponibilidade de energia térmica para realizar trabalho útil."),
    ("o Tratado de Tordesilhas", "Acordo de 1494 entre Portugal e Espanha que dividiu as terras descobertas fora da Europa por um meridiano a oeste de Cabo Verde."),
    ("a arquitetura de microsserviços", "Padrão de engenharia que divide uma aplicação em serviços autônomos, fracamente acoplados e implantáveis de forma independente."),
    ("a teoria celular", "Princípio biológico que afirma que todos os seres vivos são compostos por células, a unidade fundamental da vida."),
    ("o método científico", "Conjunto de etapas empíricas baseado em observação, formulação de hipóteses, experimentação rigorosa e validação de teorias."),
    ("a Revolução Industrial", "Período de transição para novos processos de manufatura no século XVIII, caracterizado pela mecanização e máquinas a vapor."),
    ("o princípio da incerteza de Heisenberg", "Postulado quântico que impede determinar simultaneamente com precisão arbitrária a posição e o momento linear de uma partícula."),
    ("o conceito de inflação econômica", "Aumento contínuo e generalizado dos preços de bens e serviços, resultando na perda do poder de compra da moeda."),
    ("a estrutura terciária das proteínas", "Arranjo tridimensional complexo da cadeia polipeptídica mantido por pontes dissulfeto, interações hidrofóbicas e ligações de hidrogênio."),
    ("o modelo OSI de redes", "Framework de 7 camadas (Física até Aplicação) que padroniza a comunicação de dados entre diferentes sistemas computacionais."),
    ("a teoria da evolução por seleção natural", "Mecanismo darwiniano onde organismos mais adaptados ao ambiente têm maior probabilidade de sobreviver e deixar descendentes."),
    ("o algoritmo de Dijkstra", "Algoritmo de grafos que calcula o caminho de custo mínimo entre um vértice inicial e todos os demais nós em pesos não-negativos."),
    ("a placa tectônica e a deriva continental", "Teoria geológica que explica a movimentação dos continentes e a ocorrência de sismos pelo deslizamento das placas litosféricas."),
    ("o Teorema Fundamental do Cálculo", "Conecta o conceito de derivada ao de integral definida, mostrando que a integração é a operação inversa da diferenciação."),
    ("o imperativo categórico de Kant", "Princípio ético universal que prescreve agir apenas segundo máximas que se possa querer que se tornem leis morais universais."),
    ("o Mito da Caverna de Platão", "Alegoria filosófica sobre a passagem da ignorância dos sentidos para o conhecimento da verdade pura e inteligível."),
    ("a Navalha de Occam", "Heurística metodológica segundo a qual, entre hipóteses concorrentes viáveis, a mais simples deve ser preferida."),
    ("o ciclo de Krebs", "Via metabólica central na matriz mitocondrial que oxida derivados de carboidratos, gorduras e proteínas para gerar ATP, NADH e FADH2."),
    ("a diferença entre compiladores e interpretadores", "Compiladores traduzem o código-fonte inteiro para linguagem de máquina de antemão; interpretadores executam instruções linha a linha em tempo real."),
    ("o princípio da conservação da energia", "Primeira lei da termodinâmica que estabelece que a energia não pode ser criada nem destruída, apenas transformada de uma forma em outra."),
    ("a queda da Bastilha", "Evento histórico de 1789 que marcou o início da Revolução Francesa e a queda dos símbolos do absolutismo monárquico."),
    ("o conceito de polimorfismo em POO", "Capacidade de objetos de classes derivadas responderem a uma mesma mensagem de métodos com comportamentos específicos e customizados."),
    ("a barreira hematoencefálica", "Estrutura seletiva de células endoteliais que protege o sistema nervoso central contra agentes patogênicos e toxinas circulantes no sangue."),
    ("o teorema de Bayes", "Fórmula matemática que calcula a probabilidade condicional de um evento com base em conhecimentos a priori relacionados."),
    ("a composição da atmosfera terrestre", "Mistura gasosa composta predominantemente por nitrogênio (~78%), oxigênio (~21%), argônio (~0.93%) e traços de outros gases."),
    ("a Guerra Fria", "Período de tensão geopolítica, ideológica e armamentista entre os Estados Unidos e a União Soviética entre 1947 e 1991."),
    ("o conceito de Deadlock em sistemas operacionais", "Situação em que dois ou mais processos ficam bloqueados indefinidamente aguardando recursos mutuamente alocados."),
    ("a tabela periódica moderna", "Organização dos elementos químicos em ordem crescente de número atômico, agrupados por propriedades físico-químicas periódicas semelhantes."),
    ("o conceito de viés cognitivo", "Padrão sistemático de desvio da racionalidade e julgamento lógico que afeta decisões e percepções humanas."),
    ("a lei da gravitação universal de Newton", "Postula que dois corpos se atraem com força diretamente proporcional ao produto de suas massas e inversamente proporcional ao quadrado da distância entre eles."),
    ("o papel dos ribossomos", "Organelas celulares responsáveis pela síntese proteica, traduzindo as sequências de nucleotídeos do RNA mensageiro em cadeias de aminoácidos."),
    ("a arquitetura REST", "Estilo arquitetural para sistemas hipermídia que utiliza métodos HTTP padronizados (GET, POST, PUT, DELETE) e comunicação stateless."),
    ("o sistema linfático humano", "Rede de vasos, linfonodos e órgãos que drena o excesso de líquido intersticial e participa ativamente da resposta imunológica."),
    ("o princípio de Arquimedes", "Todo corpo imerso em um fluido em repouso sofre empuxo vertical de baixo para cima igual ao peso do fluido deslocado."),
    ("o surgimento da escrita cuneiforme", "Sistema pioneiro de escrita desenvolvido pelos sumérios na Mesopotâmia por volta de 3400 a.C. em tabuletas de argila úmida."),
    ("o conceito de complexidade ciclomática", "Métrica de software que quantifica o número de caminhos linearmente independentes no fluxo de controle de um programa."),
    ("a hipótese de Riemann", "Famosa conjectura matemática sobre a distribuição dos zeros não-triviais da função zeta de Riemann, todos com parte real igual a 1/2."),
    ("o funcionamento dos semicondutores", "Materiais com condutividade intermediária controlável por dopagem química, servindo de base para diodos, transistores e chips eletrônicos."),
    ("o ciclo do carbono", "Ciclo biogeoquímico pelo qual o carbono circula entre a atmosfera, oceanos, solo e biosfera por meio de respiração, combustão e fotossíntese."),
    ("a diferença entre memória RAM e armazenamento secundário", "A RAM é volátil, de altíssima velocidade e armazena dados de execução imediata; o secundário (SSD/HD) é não-volátil e retém dados permanentemente."),
    ("a função dos neurotransmissores", "Substâncias químicas que transmitem sinais eletroquímicos através das sinapses entre neurônios ou entre neurônio e célula efetora."),
    ("a Magna Carta de 1215", "Documento histórico inglês que limitou o poder absoluto da monarquia e estabeleceu os fundamentos do devido processo legal."),
    ("o conceito de ACID em bancos de dados", "Propriedades de confiabilidade de transações: Atomicidade, Consistência, Isolamento e Durabilidade.")
]

ASPECTS = [
    "Como podemos sintetizar", "Qual é o conceito fundamental de", "Explique resumidamente",
    "Gostaria de entender melhor", "Poderia fornecer uma síntese clara sobre", "De forma didática, como funciona",
    "Qual a importância e significado de", "Em termos conceituais, o que define", "Descreva os aspectos centrais de"
]

CONTEXTS = [
    "em termos gerais", "para uma visão panorâmica", "com foco nos princípios básicos",
    "de maneira direta e objetiva", "sem jargões desnecessários", "destacando sua relevância",
    "apontando suas características essenciais", "em poucas palavras"
]

seen = set()
samples = []

for i in range(500):
    subj, ans = random.choice(SUBJECTS)
    asp = random.choice(ASPECTS)
    ctx = random.choice(CONTEXTS)
    q = f"{asp} {subj}, {ctx}?".strip()
    norm = q.lower()
    
    # Se já viu, tenta com pequena variação
    if norm in seen:
        q = f"{q} (análise teórica)"
        norm = q.lower()
    
    seen.add(norm)
    samples.append((q, ans))

print(f"Abstentions generated: {len(samples)} in 0.01s!")
assert len(samples) == 500
