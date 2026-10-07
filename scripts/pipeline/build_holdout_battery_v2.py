"""
build_holdout_battery_v2.py
Gera a nova Bateria de Avaliação Holdout (120 itens) com 50+ ferramentas e 8 domínios:
- 30 itens: Single-Tool Precisa (extração de parâmetros e tipagem estrita)
- 40 itens: Discriminação Multi-Tool (com pares de distratores finos / hard negatives)
- 30 itens: Raciocínio Algorítmico & Python (execute_python verificado em sandbox)
- 20 itens: Abstenção Negativa (ferramentas presentes, mas resposta puramente textual)

Saída: data/eval_benchmark_battery_v2.jsonl
"""

import os
import sys
import json
import math
import random
import statistics
from typing import Dict, List, Any

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_FILE = os.path.join(ROOT_DIR, "data", "eval_benchmark_battery_v2.jsonl")

# Importar o catálogo de 50 ferramentas
sys.path.insert(0, ROOT_DIR)
from data.tool_catalog.tools_schema import TOOL_CATALOG, DOMAINS, HARD_DISTRACTOR_PAIRS

random.seed(42)  # Semente fixa para reproducibilidade exata do benchmark

battery: List[Dict[str, Any]] = []

def make_item(item_id: str, level: str, category: str, system_tools: List[str], user_prompt: str, expected_tool: str, expected_args: Dict[str, Any], ground_truth_stdout: str = "", expected_text: str = "") -> Dict[str, Any]:
    tools_def = [TOOL_CATALOG[t] for t in system_tools if t in TOOL_CATALOG]
    
    # Formatação de system message
    messages = [
        {"role": "system", "content": "Você é um assistente inteligente com acesso a ferramentas. Utilize as ferramentas fornecidas quando necessário."},
        {"role": "user", "content": user_prompt}
    ]
    
    return {
        "id": item_id,
        "level": level,
        "category": category,
        "tools": tools_def,
        "tool_names": system_tools,
        "messages": messages,
        "expected_tool": expected_tool,
        "expected_args": expected_args,
        "ground_truth_stdout": ground_truth_stdout,
        "expected_text": expected_text
    }

print("=" * 70)
print("CONSTRUINDO BATERIA HOLDOUT V2 (120 ITENS)")
print("=" * 70)

# ==============================================================================
# SEÇÃO 1: SINGLE TOOL PRECISA (30 ITENS)
# ==============================================================================
single_tool_specs = [
    # DevOps
    ("docker_container_action", "Por favor, reinicie o container docker 'postgres-prod'.", {"container_id": "postgres-prod", "action": "restart"}),
    ("k8s_get_pod_status", "Consulte o status dos pods no namespace 'staging'.", {"namespace": "staging"}),
    ("check_endpoint_health", "Verifique se a URL https://api.empresa.com.br/healthz está respondendo.", {"url": "https://api.empresa.com.br/healthz"}),
    ("dns_lookup_record", "Consulte os registros MX do domínio 'acme-corp.org'.", {"domain": "acme-corp.org", "record_type": "MX"}),
    ("query_database_sql", "Execute a query 'SELECT count(*) FROM users WHERE active = 1' no banco 'analytics_db'.", {"database": "analytics_db", "sql_query": "SELECT count(*) FROM users WHERE active = 1"}),
    
    # IoT / Smart Home
    ("set_thermostat_target", "Ajuste o termostato do quarto_suite para 22 graus no modo cool.", {"room": "quarto_suite", "temperature": 22.0, "mode": "cool"}),
    ("toggle_smart_plug", "Desligue a tomada inteligente do 'ventilador'.", {"device_alias": "ventilador", "power_state": "off"}),
    ("lock_security_perimeter", "Tranque a fechadura da 'garagem'.", {"door_location": "garagem", "command": "lock"}),
    ("set_lighting_scene", "Configure a iluminação da 'sala_de_estar' para 70% com a predefinição 'sunset'.", {"zone": "sala_de_estar", "brightness_percent": 70, "color_preset": "sunset"}),
    ("open_smart_garage_door", "Abra o portão da garagem agora.", {"action": "open"}),
    
    # Produtividade
    ("create_calendar_event", "Agende a reunião 'Alinhamento Q4' para 2026-11-03T10:00:00 com duração de 45 minutos.", {"title": "Alinhamento Q4", "start_datetime": "2026-11-03T10:00:00", "duration_minutes": 45}),
    ("send_slack_notification", "Envie um alerta no canal '#deploy-alerts' com a mensagem 'Deploy v2.4 concluído com sucesso'.", {"channel": "#deploy-alerts", "message_text": "Deploy v2.4 concluído com sucesso"}),
    ("create_jira_ticket", "Abra um chamado no projeto 'PROD' do tipo 'Bug' com o resumo 'Falha na autenticação OAuth2'.", {"project_key": "PROD", "issue_type": "Bug", "summary": "Falha na autenticação OAuth2"}),
    ("search_internal_knowledge", "Pesquise na base interna corporativa sobre a 'política de reembolso de viagens'.", {"query": "política de reembolso de viagens"}),
    ("translate_document_text", "Traduza o texto 'Hello team, the quarterly report is ready.' para o português (pt).", {"target_language": "pt", "text_content": "Hello team, the quarterly report is ready."}),
    
    # Finanças
    ("get_stock_quote", "Qual a cotação atual da ação 'VALE3.SA'?", {"ticker": "VALE3.SA"}),
    ("verify_crypto_gas_price", "Verifique o gas price atual na rede 'polygon' com velocidade 'fast'.", {"network": "polygon", "speed": "fast"}),
    ("simulate_loan_amortization", "Simule um financiamento de R$ 50000 em 24 meses com taxa mensal de 1.8% no sistema 'PRICE'.", {"total_loan_amount": 50000.0, "months": 24, "interest_rate_monthly": 1.8, "system": "PRICE"}),
    ("calculate_compound_interest", "Calcule o montante de um investimento de R$ 10000 a 11.5% ao ano durante 5 anos.", {"principal": 10000.0, "annual_rate_percent": 11.5, "time_years": 5.0}),
    ("calculate_crypto_swap_rate", "Simule a troca de 2.5 ETH para USDC em DEX.", {"token_in": "ETH", "token_out": "USDC", "amount_in": 2.5}),
    
    # E-Commerce & Logística
    ("track_shipment_status", "Rastreie a encomenda com código 'NL987654321BR'.", {"tracking_code": "NL987654321BR"}),
    ("check_warehouse_inventory", "Consulte a quantidade em estoque do SKU 'NOTE-DELL-15' no centro 'CD_SP'.", {"sku_or_product_id": "NOTE-DELL-15", "warehouse_location": "CD_SP"}),
    ("apply_promotional_discount", "Aplique o cupom 'DESCONTO15' no carrinho 'cart_88412'.", {"cart_id": "cart_88412", "coupon_code": "DESCONTO15"}),
    ("request_order_refund", "Solicite o reembolso do pedido 'PED-9901' pelo motivo 'damaged_item'.", {"order_id": "PED-9901", "reason": "damaged_item"}),
    ("update_product_price", "Atualize o preço do SKU 'MOUSE-LOGI-G' para 189.90.", {"sku": "MOUSE-LOGI-G", "new_price": 189.90}),
    
    # Mídia & Utilitários
    ("generate_qr_code_matrix", "Gere um QR Code para o link 'https://meupagamento.com.br/pix/1234'.", {"data_payload": "https://meupagamento.com.br/pix/1234"}),
    ("compute_file_checksum", "Calcule o hash sha256 do arquivo '/var/data/dataset.parquet'.", {"file_path": "/var/data/dataset.parquet", "algorithm": "sha256"}),
    ("transcribe_audio_segment", "Transcreva o arquivo de áudio '/audios/entrevista_01.mp3' no idioma 'pt'.", {"audio_uri": "/audios/entrevista_01.mp3", "language": "pt"}),
    ("compress_archive_files", "Compacte os arquivos ['relatorio.pdf', 'dados.csv'] no arquivo 'relatorio_final.zip' formato zip.", {"source_paths": ["relatorio.pdf", "dados.csv"], "output_archive_name": "relatorio_final.zip"}),
    ("extract_pdf_pages", "Extraia as páginas '1-3' do arquivo 'manual_completo.pdf' para 'manual_resumo.pdf'.", {"input_pdf_path": "manual_completo.pdf", "page_range": "1-3", "output_pdf_path": "manual_resumo.pdf"})
]

for idx, (t_name, prompt, args) in enumerate(single_tool_specs, 1):
    battery.append(make_item(
        item_id=f"holdout_v2_L1_{idx:03d}",
        level="level_1_foundation",
        category="single_tool_precise",
        system_tools=[t_name],
        user_prompt=prompt,
        expected_tool=t_name,
        expected_args=args
    ))

print(f"Seção 1 (Single Tool): {len(battery)} itens criados.")

# ==============================================================================
# SEÇÃO 2: DISCRIMINAÇÃO MULTI-TOOL COM HARD NEGATIVES (40 ITENS)
# ==============================================================================
multi_disc_specs = [
    # 1. Moeda: Cotação unitária vs Conversão com montante
    (["get_exchange_rate", "convert_currency_amount", "get_stock_quote"],
     "Quanto está a taxa de câmbio de 1 USD para BRL hoje?",
     "get_exchange_rate", {"from_currency": "USD", "to_currency": "BRL"}),
     
    (["get_exchange_rate", "convert_currency_amount", "get_stock_quote"],
     "Converta 350 dólares para reais com a cotação de hoje.",
     "convert_currency_amount", {"amount": 350.0, "from_currency": "USD", "to_currency": "BRL"}),

    # 2. Imagem: Redimensionar vs Converter formato
    (["resize_image_dimensions", "convert_image_format", "compute_file_checksum"],
     "Redimensione o banner 'assets/topo.png' para 1200x400 pixels.",
     "resize_image_dimensions", {"image_path": "assets/topo.png", "width": 1200, "height": 400}),

    (["resize_image_dimensions", "convert_image_format", "compute_file_checksum"],
     "Converta a foto 'fotos/perfil.png' para o formato webp com qualidade 90.",
     "convert_image_format", {"source_image": "fotos/perfil.png", "target_format": "webp", "quality": 90}),

    # 3. Logística: Rastreio vs Frete
    (["track_shipment_status", "calculate_shipping_rates", "check_warehouse_inventory"],
     "Onde está a minha encomenda com o código 'BR445566778SP'?",
     "track_shipment_status", {"tracking_code": "BR445566778SP"}),

    (["track_shipment_status", "calculate_shipping_rates", "check_warehouse_inventory"],
     "Calcule quanto custa o frete para enviar um pacote de 3.5 kg do CEP 01310100 para o CEP 20040002.",
     "calculate_shipping_rates", {"origin_zip": "01310100", "destination_zip": "20040002", "weight_kg": 3.5}),

    # 4. Infra: Docker vs Kubernetes
    (["docker_container_action", "k8s_get_pod_status", "check_endpoint_health"],
     "Pare imediatamente o container Docker 'api-worker-3'.",
     "docker_container_action", {"container_id": "api-worker-3", "action": "stop"}),

    (["docker_container_action", "k8s_get_pod_status", "check_endpoint_health"],
     "Verifique a saúde e status dos pods do Kubernetes no namespace 'production'.",
     "k8s_get_pod_status", {"namespace": "production"}),

    # 5. Comunicação: Slack vs Email
    (["send_slack_notification", "draft_email_reply", "create_jira_ticket"],
     "Avise o time no canal '#backend' que o servidor passará por manutenção às 23h.",
     "send_slack_notification", {"channel": "#backend", "message_text": "O servidor passará por manutenção às 23h"}),

    (["send_slack_notification", "draft_email_reply", "create_jira_ticket"],
     "Redija uma resposta de email para 'cliente@acme.com' com o assunto 'Proposta Comercial' dizendo que enviamos a proposta anexa.",
     "draft_email_reply", {"recipient": "cliente@acme.com", "subject": "Proposta Comercial", "body_markdown": "Enviamos a proposta anexa."}),

    # 6. Mercado: Ações (cotação) vs Demonstrativo da empresa
    (["get_stock_quote", "get_company_financials", "calculate_compound_interest"],
     "Veja o preço atual de negociação das ações da 'AAPL'.",
     "get_stock_quote", {"ticker": "AAPL"}),

    (["get_stock_quote", "get_company_financials", "calculate_compound_interest"],
     "Consulte o demonstrativo de resultado 'income_statement' da empresa 'PETR4.SA' para o ano 2025.",
     "get_company_financials", {"ticker": "PETR4.SA", "statement_type": "income_statement", "fiscal_year": 2025}),

    # 7. Calendário vs Fuso Horário
    (["create_calendar_event", "get_timezone_time", "send_slack_notification"],
     "Que horas são exatamente agora no fuso horário 'America/New_York'?",
     "get_timezone_time", {"timezone": "America/New_York"}),

    (["create_calendar_event", "get_timezone_time", "send_slack_notification"],
     "Marque uma reunião no calendário chamada 'Review de Sprint' em 2026-10-20T15:00:00 com 60 minutos.",
     "create_calendar_event", {"title": "Review de Sprint", "start_datetime": "2026-10-20T15:00:00", "duration_minutes": 60}),

    # 8. Unidades Físicas vs Unidades Monetárias
    (["convert_measurement_units", "convert_currency_amount", "calculate_compound_interest"],
     "Converta uma distância de 42 km para milhas.",
     "convert_measurement_units", {"value": 42.0, "from_unit": "km", "to_unit": "milhas"}),

    (["convert_measurement_units", "convert_currency_amount", "calculate_compound_interest"],
     "Converta 1200 euros para dólares (EUR para USD).",
     "convert_currency_amount", {"amount": 1200.0, "from_currency": "EUR", "to_currency": "USD"}),

    # 9. CloudWatch vs Endpoint Health vs DNS
    (["query_cloudwatch_metrics", "check_endpoint_health", "dns_lookup_record"],
     "Consulte o histórico de uso de CPU do serviço AWS 'AWS/RDS' nos últimos 30 minutos.",
     "query_cloudwatch_metrics", {"service_name": "AWS/RDS", "metric_name": "CPUUtilization", "period_minutes": 30}),

    (["query_cloudwatch_metrics", "check_endpoint_health", "dns_lookup_record"],
     "Faça um teste de healthcheck para ver se 'https://checkout.loja.com' está online.",
     "check_endpoint_health", {"url": "https://checkout.loja.com"}),

    (["query_cloudwatch_metrics", "check_endpoint_health", "dns_lookup_record"],
     "Verifique o registro IP tipo 'A' do domínio 'cloudflare.com'.",
     "dns_lookup_record", {"domain": "cloudflare.com", "record_type": "A"}),

    # 10. Temperatura Residencial vs Clima Externo
    (["set_thermostat_target", "get_current_weather", "set_lighting_scene"],
     "Mude a temperatura do termostato da 'sala' para 23 graus no modo 'eco'.",
     "set_thermostat_target", {"room": "sala", "temperature": 23.0, "mode": "eco"}),

    (["set_thermostat_target", "get_current_weather", "set_lighting_scene"],
     "Como está o clima e a temperatura atual lá fora na cidade de 'Curitiba' em celsius?",
     "get_current_weather", {"city": "Curitiba", "unit": "celsius"}),

    # 11. Armazém vs Atualização de Preço vs Cupom
    (["check_warehouse_inventory", "update_product_price", "apply_promotional_discount"],
     "Quantas unidades da placa de vídeo 'GPU-RTX-4080' temos no armazém 'CD_MG'?",
     "check_warehouse_inventory", {"sku_or_product_id": "GPU-RTX-4080", "warehouse_location": "CD_MG"}),

    (["check_warehouse_inventory", "update_product_price", "apply_promotional_discount"],
     "Altere o preço de tabela do item 'TECLADO-MEC' para 320.00.",
     "update_product_price", {"sku": "TECLADO-MEC", "new_price": 320.00}),

    # 12. Finanças: Juros Compostos vs Amortização de Empréstimo
    (["calculate_compound_interest", "simulate_loan_amortization", "get_stock_quote"],
     "Se eu aplicar 20000 reais a 10% ao ano por 3 anos com aporte mensal de 500 reais, qual será o montante?",
     "calculate_compound_interest", {"principal": 20000.0, "annual_rate_percent": 10.0, "time_years": 3.0, "monthly_contribution": 500.0}),

    (["calculate_compound_interest", "simulate_loan_amortization", "get_stock_quote"],
     "Calcule a tabela SAC para um empréstimo de 80000 reais parcelado em 36 meses com taxa de 1.5% ao mês.",
     "simulate_loan_amortization", {"total_loan_amount": 80000.0, "months": 36, "interest_rate_monthly": 1.5, "system": "SAC"}),

    # 13. Cripto: Gas Price vs Swap DEX
    (["verify_crypto_gas_price", "calculate_crypto_swap_rate", "convert_currency_amount"],
     "Como está a taxa de gas na rede 'ethereum' para transações em velocidade 'standard'?",
     "verify_crypto_gas_price", {"network": "ethereum", "speed": "standard"}),

    (["verify_crypto_gas_price", "calculate_crypto_swap_rate", "convert_currency_amount"],
     "Estime o retorno ao trocar 500 USDT por SOL em exchange descentralizada.",
     "calculate_crypto_swap_rate", {"token_in": "USDT", "token_out": "SOL", "amount_in": 500.0}),

    # 14. Arquivos: Compressão vs Checksum vs PDF
    (["compress_archive_files", "compute_file_checksum", "extract_pdf_pages"],
     "Gere o hash md5 de verificação do instalador '/downloads/setup.exe'.",
     "compute_file_checksum", {"file_path": "/downloads/setup.exe", "algorithm": "md5"}),

    (["compress_archive_files", "compute_file_checksum", "extract_pdf_pages"],
     "Extraia as páginas '4-8' do documento '/docs/relatorio.pdf' para '/docs/capitulo2.pdf'.",
     "extract_pdf_pages", {"input_pdf_path": "/docs/relatorio.pdf", "page_range": "4-8", "output_pdf_path": "/docs/capitulo2.pdf"}),

    (["compress_archive_files", "compute_file_checksum", "extract_pdf_pages"],
     "Compacte a pasta ['/logs/app1', '/logs/app2'] no arquivo de saída 'logs_backup.tar.gz' formato tar.gz.",
     "compress_archive_files", {"source_paths": ["/logs/app1", "/logs/app2"], "output_archive_name": "logs_backup.tar.gz", "format": "tar.gz"}),

    # 15. Smart Home: Tranca vs Portão da Garagem vs Câmera
    (["lock_security_perimeter", "open_smart_garage_door", "get_security_camera_snapshot"],
     "Abra o portão eletrônico da garagem para a entrega.",
     "open_smart_garage_door", {"action": "open"}),

    (["lock_security_perimeter", "open_smart_garage_door", "get_security_camera_snapshot"],
     "Destranque a fechadura eletrônica da 'porta_principal'.",
     "lock_security_perimeter", {"door_location": "porta_principal", "command": "unlock"}),

    (["lock_security_perimeter", "open_smart_garage_door", "get_security_camera_snapshot"],
     "Veja a imagem atual da câmera 'cam_quintal' com detecção de pessoas ativada.",
     "get_security_camera_snapshot", {"camera_id": "cam_quintal", "include_ai_detection": True}),

    # 16. IoT: Tomada vs Consumo de Energia
    (["toggle_smart_plug", "get_power_consumption_kwh", "set_lighting_scene"],
     "Ligue a tomada inteligente da 'cafeteira'.",
     "toggle_smart_plug", {"device_alias": "cafeteira", "power_state": "on"}),

    (["toggle_smart_plug", "get_power_consumption_kwh", "set_lighting_scene"],
     "Qual foi o consumo em kWh do circuito 'ar_condicionado' no período 'current_month'?",
     "get_power_consumption_kwh", {"circuit_name": "ar_condicionado", "timeframe": "current_month"}),

    # 17. CI/CD vs S3 Bucket
    (["trigger_ci_build", "s3_bucket_operation", "docker_container_action"],
     "Inicie a execução da pipeline de CI no repositório 'empresa/pagamentos' na branch 'develop'.",
     "trigger_ci_build", {"repository": "empresa/pagamentos", "branch": "develop"}),

    (["trigger_ci_build", "s3_bucket_operation", "docker_container_action"],
     "Liste os objetos armazenados dentro do bucket S3 'empresa-backups-2026'.",
     "s3_bucket_operation", {"bucket_name": "empresa-backups-2026", "operation": "list_objects"}),

    # 18. Pesquisa Interna vs Busca de Notícias
    (["search_internal_knowledge", "search_web_news", "summarize_email_thread"],
     "Busque notícias recentes na web aberta sobre 'inteligência artificial em semicondutores'.",
     "search_web_news", {"query": "inteligência artificial em semicondutores"}),

    (["search_internal_knowledge", "search_web_news", "summarize_email_thread"],
     "Procure na nossa documentação interna da empresa sobre 'como configurar o VPN corporativo'.",
     "search_internal_knowledge", {"query": "como configurar o VPN corporativo"}),

    # 19. Tradução vs Resumo de Email
    (["translate_document_text", "summarize_email_thread", "draft_email_reply"],
     "Traduza a frase 'The system architecture has been updated.' para francês (fr).",
     "translate_document_text", {"target_language": "fr", "text_content": "The system architecture has been updated."}),

    (["translate_document_text", "summarize_email_thread", "draft_email_reply"],
     "Resuma os principais pontos discutidos na thread de email com id 'thread_projeto_alfa_99'.",
     "summarize_email_thread", {"thread_id": "thread_projeto_alfa_99"})
]

for idx, (tools, prompt, exp_tool, exp_args) in enumerate(multi_disc_specs, 1):
    battery.append(make_item(
        item_id=f"holdout_v2_L2_{idx:03d}",
        level="level_2_discrimination",
        category="multi_tool_discrimination",
        system_tools=tools,
        user_prompt=prompt,
        expected_tool=exp_tool,
        expected_args=exp_args
    ))

print(f"Seção 2 (Discriminação Multi-Tool): {len(battery) - 30} itens criados.")

# ==============================================================================
# SEÇÃO 3: RACIOCÍNIO ALGORÍTMICO & PYTHON (30 ITENS CERTIFICADOS)
# ==============================================================================
math_problems = [
    # 1. Álgebra e Séries
    ("Calcule a soma dos quadrados dos primeiros 25 números inteiros positivos.",
     "res = sum(i**2 for i in range(1, 26))\nprint(res)"),
     
    ("Determine a soma de todos os múltiplos de 7 ou 11 estritamente menores que 500.",
     "res = sum(x for x in range(500) if x % 7 == 0 or x % 11 == 0)\nprint(res)"),

    ("Qual é o 35º termo da sequência de Fibonacci (onde F(1)=1, F(2)=1)?",
     "a, b = 1, 1\nfor _ in range(33):\n    a, b = b, a + b\nprint(b)"),

    # 2. Teoria dos Números e Primos
    ("Quantos números primos existem no intervalo fechado de 100 a 300?",
     "def is_p(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True\nres = sum(1 for x in range(100, 301) if is_p(x))\nprint(res)"),

    ("Encontre a soma de todos os divisores próprios positivos de 2520.",
     "n = 2520\nres = sum(i for i in range(1, n) if n % i == 0)\nprint(res)"),

    ("Qual é o resto da divisão de 3^2025 por 1000?",
     "res = pow(3, 2025, 1000)\nprint(res)"),

    ("Determine o Máximo Divisor Comum (MDC) entre os números 4824 e 3564.",
     "import math\nprint(math.gcd(4824, 3564))"),

    ("Determine o Mínimo Múltiplo Comum (MMC) entre 84, 126 e 210.",
     "import math\ndef lcm(a, b):\n    return abs(a*b) // math.gcd(a, b)\nprint(lcm(lcm(84, 126), 210))"),

    # 3. Combinatória e Probabilidade
    ("De quantas maneiras distintas podemos formar uma comissão de 5 pessoas a partir de um grupo de 14 pessoas?",
     "import math\nprint(math.comb(14, 5))"),

    ("Quantos anagramas distintos podem ser formados com as letras da palavra 'PARALELEPIPEDO'?",
     "import math\nfrom collections import Counter\nword = 'PARALELEPIPEDO'\ntotal = math.factorial(len(word))\ncounts = Counter(word)\ndenom = 1\nfor c in counts.values():\n    denom *= math.factorial(c)\nprint(total // denom)"),

    ("Quantos números de 4 dígitos distintos podem ser formados usando apenas os algarismos {1, 2, 3, 4, 5, 6, 7}?",
     "import math\nprint(math.perm(7, 4))"),

    # 4. Geometria e Física Computacional
    ("Calcule a hipotenusa de um triângulo retângulo com catetos de comprimento 45 e 60.",
     "import math\nprint(round(math.hypot(45, 60), 2))"),

    ("Calcule a área exata de um trapézio com bases de comprimento 24 e 16, e altura medindo 11.5.",
     "b1, b2, h = 24, 16, 11.5\narea = ((b1 + b2) * h) / 2\nprint(area)"),

    ("Um corpo é lançado verticalmente com velocidade inicial de 49 m/s. Considerando g = 9.8 m/s^2, qual a altura máxima atingida?",
     "v0 = 49.0\ng = 9.8\nh_max = (v0**2) / (2 * g)\nprint(round(h_max, 2))"),

    # 5. Estatística e Vetores
    ("Calcule a variância amostral dos seguintes dados: [14, 18, 22, 25, 29, 31, 35, 42].",
     "import statistics\ndata = [14, 18, 22, 25, 29, 31, 35, 42]\nprint(round(statistics.variance(data), 2))"),

    ("Calcule a média geométrica dos valores [2, 4, 8, 16, 32, 64].",
     "import statistics\ndata = [2, 4, 8, 16, 32, 64]\nprint(round(statistics.geometric_mean(data), 2))"),

    ("Calcule o produto escalar (dot product) entre os vetores u = [3, -2, 5, 4] e v = [7, 6, -1, 3].",
     "u = [3, -2, 5, 4]\nv = [7, 6, -1, 3]\nprint(sum(x*y for x, y in zip(u, v)))"),

    # 6. Matrizes
    ("Calcule o determinante da matriz 3x3: [[4, 2, -1], [3, 0, 5], [-2, 1, 2]].",
     "m = [[4, 2, -1], [3, 0, 5], [-2, 1, 2]]\ndet = (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))\nprint(det)"),

    ("Calcule o traço (soma da diagonal principal) da matriz: [[12, 5, 3], [8, 19, -4], [0, 7, 31]].",
     "m = [[12, 5, 3], [8, 19, -4], [0, 7, 31]]\nprint(sum(m[i][i] for i in range(3)))"),

    # 7. Problemas Verbais de Aritmética
    ("Uma loja dá 20% de desconto em um produto de R$ 450,00 e depois cobra 8% de imposto sobre o valor com desconto. Qual o preço final?",
     "preco = 450.0\ncom_desc = preco * 0.8\nfinal = com_desc * 1.08\nprint(round(final, 2))"),

    ("Em uma sala de aula, a razão entre meninas e meninos é de 4 para 3. Se há 28 meninas, quantos alunos há no total na sala?",
     "meninas = 28\nmeninos = (meninas * 3) // 4\ntotal = meninas + meninos\nprint(total)"),

    ("Um ciclista percorre 75 km em 2 horas e 30 minutos. Qual é a sua velocidade média em km/h?",
     "dist = 75.0\ntempo = 2.5\nprint(round(dist / tempo, 2))"),

    # 8. Computação de Strings e Algoritmos
    ("Quantas vogais (a, e, i, o, u) existem no texto: 'Supercomputadores realizam calculos cientificos complexos em paralelo'?",
     "t = 'Supercomputadores realizam calculos cientificos complexos em paralelo'.lower()\nvogais = set('aeiou')\nprint(sum(1 for c in t if c in vogais))"),

    ("Qual é o valor em base 10 do número binário '110101101'?",
     "print(int('110101101', 2))"),

    ("Converta o número decimal 1459 para sua representação hexadecimal.",
     "print(hex(1459)[2:].upper())"),

    # 9. Potências e Raízes
    ("Calcule o valor de (17^3) + (19^3) - (13^3).",
     "print((17**3) + (19**3) - (13**3))"),

    ("Qual é o valor inteiro mais próximo da raiz cúbica de 100000?",
     "print(round(100000**(1/3)))"),

    ("Calcule a soma de todos os dígitos do número 2^100.",
     "print(sum(int(d) for d in str(2**100)))"),

    # 10. Problemas Lógicos Numéricos
    ("Qual é o menor número inteiro positivo divisível por todos os números de 1 a 12?",
     "import math\ndef lcm(a, b):\n    return abs(a*b) // math.gcd(a, b)\nres = 1\nfor i in range(1, 13):\n    res = lcm(res, i)\nprint(res)"),

    ("Encontre o 100º número par positivo.",
     "print(100 * 2)")
]

for idx, (prompt, code) in enumerate(math_problems, 1):
    # Execução isolada em Python para obter o stdout verificado
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"math": math, "statistics": statistics})
        stdout_val = buf.getvalue().strip()
    except Exception as e:
        print(f"Erro ao certificar problema {idx}: {e}")
        stdout_val = "ERROR"

    battery.append(make_item(
        item_id=f"holdout_v2_L3_{idx:03d}",
        level="level_3_reasoning_python",
        category="python_algorithmic_reasoning",
        system_tools=["execute_python"],
        user_prompt=prompt,
        expected_tool="execute_python",
        expected_args={"code": code},
        ground_truth_stdout=stdout_val
    ))

print(f"Seção 3 (Python Math): {len(battery) - 70} itens certificados e validados.")

# ==============================================================================
# SEÇÃO 4: ABSTENÇÃO NEGATIVA (20 ITENS)
# ==============================================================================
abstention_specs = [
    (["docker_container_action", "k8s_get_pod_status", "check_endpoint_health"],
     "O que é uma variável de ambiente em sistemas operacionais e para que ela serve?",
     "Uma variável de ambiente é um valor dinâmico que o sistema operacional disponibiliza para processos e programas em execução."),

    (["set_thermostat_target", "toggle_smart_plug", "set_lighting_scene"],
     "Quem pintou a Mona Lisa e em qual museu ela está exposta?",
     "A Mona Lisa foi pintada por Leonardo da Vinci e está exposta no Museu do Louvre, em Paris."),

    (["create_calendar_event", "send_slack_notification", "create_jira_ticket"],
     "Qual é a diferença entre a metodologia Ágil Scrum e o método Kanban?",
     "O Scrum trabalha com sprints fixas de tempo e papéis bem definidos, enquanto o Kanban é um fluxo contínuo focado em limitar o trabalho em progresso (WIP)."),

    (["convert_currency_amount", "get_stock_quote", "calculate_compound_interest"],
     "O que significa o conceito de 'Custo de Oportunidade' na economia?",
     "Custo de oportunidade é o benefício ou valor que se abre mão ao escolher uma alternativa em detrimento de outra."),

    (["track_shipment_status", "check_warehouse_inventory", "calculate_shipping_rates"],
     "Explique brevemente a teoria da relatividade geral de Albert Einstein.",
     "A relatividade geral postula que a gravidade não é uma força tradicional, mas sim a curvatura do espaço-tempo provocada pela presença de massa e energia."),

    (["resize_image_dimensions", "generate_qr_code_matrix", "compute_file_checksum"],
     "Como funciona o algoritmo de busca binária e qual a sua complexidade de tempo?",
     "A busca binária divide um array ordenado repetidamente pela metade, alcançando complexidade de tempo logarítmica O(log n)."),

    (["get_current_weather", "search_web_news", "convert_measurement_units"],
     "Qual é a capital do Canadá?",
     "A capital do Canadá é Ottawa."),

    (["docker_container_action", "s3_bucket_operation", "trigger_ci_build"],
     "Quais são os quatro pilares da Programação Orientada a Objetos?",
     "Os quatro pilares da POO são: Abstração, Encapsulamento, Herança e Polimorfismo."),

    (["open_smart_garage_door", "lock_security_perimeter", "get_power_consumption_kwh"],
     "Como ocorre o processo de fotossíntese nas plantas?",
     "A fotossíntese converte energia solar, água e dióxido de carbono em glicose e oxigênio por meio da clorofila nas células vegetais."),

    (["draft_email_reply", "summarize_email_thread", "translate_document_text"],
     "Quem escreveu a obra 'Dom Casmurro' e em que ano foi publicada?",
     "Dom Casmurro foi escrito por Machado de Assis e publicado no ano de 1899."),

    (["get_company_financials", "simulate_loan_amortization", "verify_crypto_gas_price"],
     "O que é uma 'Árvore Binária de Busca' (BST)?",
     "Uma BST é uma estrutura de dados em árvore onde cada nó possui no máximo dois filhos, com os nós à esquerda menores e à direita maiores que o nó raiz."),

    (["apply_promotional_discount", "request_order_refund", "generate_customer_invoice_pdf"],
     "Qual é o rio mais longo do mundo em extensão territorial?",
     "O Rio Amazonas é considerado o rio mais longo e com maior volume de água do mundo."),

    (["transcribe_audio_segment", "compress_archive_files", "convert_image_format"],
     "O que são números primos gêmeos na matemática?",
     "Primos gêmeos são pares de números primos que possuem uma diferença de exatamente dois valores entre si (como 3 e 5, ou 11 e 13)."),

    (["check_endpoint_health", "dns_lookup_record", "query_database_sql"],
     "Qual a diferença conceitual entre TCP e UDP em redes de computadores?",
     "O TCP é orientado a conexão e garante a entrega ordenada de pacotes, enquanto o UDP é não orientado a conexão, priorizando velocidade sem garantia de retransmissão."),

    (["toggle_smart_plug", "set_lighting_scene", "get_security_camera_snapshot"],
     "Em que ano o ser humano pisou pela primeira vez na Lua?",
     "O ser humano pisou na Lua em 20 de julho de 1969, na missão Apollo 11."),

    (["send_slack_notification", "search_internal_knowledge", "create_jira_ticket"],
     "O que caracteriza o princípio da navalha de Occam?",
     "A navalha de Occam diz que, entre hipóteses concorrentes que explicam o mesmo fenômeno, a mais simples (com menos suposições) tende a ser a mais correta."),

    (["convert_currency_amount", "get_exchange_rate", "calculate_crypto_swap_rate"],
     "Explique o que é a inflação e quais suas causas primárias.",
     "Inflação é o aumento generalizado e contínuo dos preços de bens e serviços em uma economia, geralmente causada por aumento de demanda ou expansão da oferta monetária."),

    (["track_shipment_status", "update_product_price", "check_warehouse_inventory"],
     "Quais são as três leis do movimento de Isaac Newton?",
     "As três leis são: Lei da Inércia, Princípio Fundamental da Dinâmica (F = m*a) e Lei da Ação e Reação."),

    (["resize_image_dimensions", "extract_pdf_pages", "generate_qr_code_matrix"],
     "O que é a Tabela Periódica dos Elementos Químicos e quem a organizou inicialmente?",
     "A Tabela Periódica organiza os elementos químicos por número atômico e propriedades periódicas, proposta inicialmente por Dmitri Mendeleev em 1869."),

    (["calculate_flight_duration", "get_timezone_time", "convert_measurement_units"],
     "Por que o céu é azul durante o dia?",
     "O céu é azul devido ao espalhamento Rayleigh, onde as moléculas da atmosfera espalham comprimentos de onda curtos (luz azul) em maior intensidade.")
]

for idx, (tools, prompt, text) in enumerate(abstention_specs, 1):
    battery.append(make_item(
        item_id=f"holdout_v2_L4_{idx:03d}",
        level="level_1_foundation",
        category="negative_abstention",
        system_tools=tools,
        user_prompt=prompt,
        expected_tool="",  # Nenhuma ferramenta esperada
        expected_args={},
        expected_text=text
    ))

print(f"Seção 4 (Abstenção Negativa): {len(battery) - 100} itens criados.")

print("=" * 70)
print(f"TOTAL DE ITENS NA BATERIA HOLDOUT V2: {len(battery)}")
print(f"Salvando em: {OUTPUT_FILE}...")
print("=" * 70)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in battery:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Bateria Holdout v2 gerada com sucesso e selada!")
