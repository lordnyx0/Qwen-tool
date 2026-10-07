import random
import time

CITIES = [
    "Tóquio", "Londres", "Paris", "Berlim", "Roma", "Madrid", "Lisboa", "Toronto", "Sydney",
    "São Paulo", "Rio de Janeiro", "Curitiba", "Seul", "Singapura", "Estocolmo", "Oslo",
    "Amsterdã", "Viena", "Zurique", "Santiago", "Buenos Aires", "Bogotá", "Lima", "Cidade do México",
    "Nova York", "Los Angeles", "Chicago", "Miami", "San Francisco", "Vancouver", "Montreal",
    "Dublin", "Copenhague", "Helsinque", "Bruxelas", "Praga", "Varsóvia", "Budapeste", "Atenas",
    "Istambul", "Cairo", "Joanesburgo", "Nairóbi", "Dubai", "Doha", "Pequim", "Xangai",
    "Bangcoc", "Kuala Lumpur", "Jacarta", "Manila", "Auckland", "Brasília", "Belo Horizonte", "Porto Alegre"
]

CURRENCIES = ["USD", "BRL", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "MXN", "CLP", "ARS", "NZD", "SEK", "NOK", "SGD"]

TIMEZONES = [
    "America/Sao_Paulo", "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
    "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Rome", "Europe/Madrid", "Europe/Lisbon",
    "Asia/Tokyo", "Asia/Shanghai", "Asia/Singapore", "Asia/Dubai", "Asia/Hong_Kong", "Asia/Seoul",
    "Australia/Sydney", "Australia/Melbourne", "Pacific/Auckland", "Africa/Johannesburg", "America/Toronto"
]

UNIT_PAIRS = [
    ("quilômetros", "milhas", 10, 5000),
    ("metros", "pés", 5, 2000),
    ("quilogramas", "libras", 1, 1000),
    ("gramas", "onças", 10, 2000),
    ("litros", "galões", 2, 500),
    ("mililitros", "onças fluidas", 50, 1500),
    ("polegadas", "centímetros", 2, 500)
]

SEARCH_TOPICS = [
    "Docker compose networking", "PostgreSQL indexing btree", "Python asyncio event loop",
    "Kubernetes ingress controller", "Redis eviction policies", "Git rebase interactive",
    "Linux systemd service configuration", "OAuth2 authorization code flow", "JWT token expiration",
    "FastAPI dependency injection", "PyTorch gradient checkpointing", "Transformers attention mask",
    "Nginx reverse proxy ssl", "CSS flexbox align items", "TypeScript utility types",
    "Rust ownership borrow checker", "Golang goroutines channels", "Terraform state locking",
    "Prometheus metrics scraping", "Kafka partition key hashing", "SQLite WAL mode",
    "GraphQL schema resolvers", "Elasticsearch fuzzy query", "MongoDB aggregation pipeline",
    "RabbitMQ dead letter exchange", "Apache Spark dataframe join", "Ansible playbook vault",
    "WebAssembly memory buffer", "gRPC protobuf streaming", "OpenTelemetry distributed tracing"
]

t0 = time.time()
seen = set()
single_samples = []

while len(single_samples) < 500:
    t_type = random.randint(0, 4)
    if t_type == 0:
        c = random.choice(CITIES)
        unit = random.choice(["celsius", "fahrenheit"])
        q = random.choice([
            f"Qual é a temperatura atual em {c} em graus {unit}?",
            f"Como está o clima em {c} agora? Use {unit}.",
            f"Verifique as condições meteorológicas em tempo real para {c} ({unit}).",
            f"Me informe a previsão do tempo para a cidade de {c} em {unit}.",
            f"Consulte o clima atual de {c} expressando a medição em graus {unit}."
        ])
        fn_name = "get_current_weather"
        args = {"city": c, "unit": unit}
    elif t_type == 1:
        fc = random.choice(CURRENCIES)
        tc = random.choice([x for x in CURRENCIES if x != fc])
        amt = random.randint(15, 95000)
        q = random.choice([
            f"Converta {amt} {fc} para {tc} na cotação comercial de hoje.",
            f"Quanto vale a quantia de {amt} {fc} convertidos em {tc}?",
            f"Faça o câmbio monetário de {amt} {fc} para a moeda {tc}.",
            f"Qual é o valor exato de {amt} {fc} expresso em {tc}?"
        ])
        fn_name = "convert_currency"
        args = {"amount": amt, "from_currency": fc, "to_currency": tc}
    elif t_type == 2:
        tz = random.choice(TIMEZONES)
        q = random.choice([
            f"Que horas são neste momento no fuso horário {tz}?",
            f"Informe a hora oficial para o fuso {tz}.",
            f"Qual o horário exato registrado em {tz} agora?",
            f"Consulte o horário corrente no relógio de {tz}."
        ])
        fn_name = "get_timezone_time"
        args = {"timezone": tz}
    elif t_type == 3:
        u1, u2, vmin, vmax = random.choice(UNIT_PAIRS)
        val = random.randint(vmin, vmax)
        q = random.choice([
            f"Converta a medida de {val} {u1} para {u2}.",
            f"Quanto corresponde {val} {u1} convertidos em {u2}?",
            f"Transforme o valor físico de {val} {u1} para a unidade {u2}."
        ])
        fn_name = "convert_units"
        args = {"value": val, "from_unit": u1, "to_unit": u2}
    else:
        topic = random.choice(SEARCH_TOPICS)
        sub = random.choice(["guia prático", "melhores práticas de produção", "resolução de problemas comuns", "documentação de arquitetura", "passo a passo para iniciantes"])
        uid = random.randint(100, 9999)
        q = random.choice([
            f"Pesquise na base de conhecimento sobre '{topic}' com foco em {sub} (ID #{uid}).",
            f"Consulte na documentação interna o artigo a respeito de '{topic}' ({sub}, ref #{uid}).",
            f"Busque nos manuais técnicos por informações sobre '{topic}' - {sub} [registro #{uid}]."
        ])
        fn_name = "search_knowledge_base"
        args = {"query": f"{topic} {sub}"}

    norm = q.lower()
    if norm not in seen:
        seen.add(norm)
        single_samples.append((q, fn_name, args))

elapsed = time.time() - t0
print(f"Single tool samples generated: {len(single_samples)} in {elapsed:.3f}s!")
