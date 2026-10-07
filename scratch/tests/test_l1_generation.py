import random
import json
import re

random.seed(42)

def test_l1_generation():
    CITIES = [
        "Tóquio", "Londres", "Paris", "Berlim", "Roma", "Madrid", "Lisboa", "Toronto", "Sydney",
        "São Paulo", "Rio de Janeiro", "Curitiba", "Seul", "Singapura", "Estocolmo", "Oslo",
        "Amsterdã", "Viena", "Zurique", "Santiago", "Buenos Aires", "Bogotá", "Lima", "Cidade do México",
        "Nova York", "Los Angeles", "Chicago", "Miami", "San Francisco", "Vancouver", "Montreal",
        "Dublin", "Copenhague", "Helsinque", "Bruxelas", "Praga", "Varsóvia", "Budapeste", "Atenas",
        "Istambul", "Cairo", "Joanesburgo", "Nairóbi", "Dubai", "Doha", "Tóquio", "Pequim", "Xangai",
        "Bangcoc", "Kuala Lumpur", "Jacarta", "Manila", "Auckland", "Brasília", "Belo Horizonte", "Porto Alegre"
    ]
    
    CURRENCIES = [
        "USD", "BRL", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "MXN", "CLP", "ARS", "NZD", "SEK", "NOK", "SGD"
    ]

    TIMEZONES = [
        "America/Sao_Paulo", "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
        "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Rome", "Europe/Madrid", "Europe/Lisbon",
        "Asia/Tokyo", "Asia/Shanghai", "Asia/Singapore", "Asia/Dubai", "Asia/Hong_Kong", "Asia/Seoul",
        "Australia/Sydney", "Australia/Melbourne", "Pacific/Auckland", "Africa/Johannesburg", "America/Toronto"
    ]

    UNIT_PAIRS = [
        ("quilômetros", "milhas", "convert_length", 10, 5000),
        ("metros", "pés", "convert_length", 5, 2000),
        ("quilogramas", "libras", "convert_weight", 1, 1000),
        ("gramas", "onças", "convert_weight", 10, 2000),
        ("litros", "galões", "convert_volume", 2, 500),
        ("mililitros", "onças fluidas", "convert_volume", 50, 1500),
        ("graus celsius", "fahrenheit", "convert_temperature", -20, 50),
    ]

    SEARCH_TOPICS = [
        "Docker compose networking", "PostgreSQL indexing btree", "Python asyncio event loop",
        "Kubernetes ingress controller", "Redis eviction policies", "Git rebase interactive",
        "Linux systemd service configuration", "OAuth2 authorization code flow", "JWT token expiration",
        "FastAPI dependency injection", "PyTorch gradient checkpointing", "Transformers attention mask",
        "Nginx reverse proxy ssl", "CSS flexbox align items", "TypeScript utility types",
        "Rust ownership borrow checker", "Golang goroutines channels", "Terraform state locking",
        "Prometheus metrics scraping", "Kafka partition key hashing", "SQLite WAL mode",
        "GraphQL schema resolvers", "Elasticsearch fuzzy query", "MongoDB aggregation pipeline"
    ]

    seen = set()
    single_samples = []

    for i in range(500):
        t_type = i % 5
        for attempt in range(100):
            if t_type == 0:
                c = random.choice(CITIES)
                unit = random.choice(["celsius", "fahrenheit"])
                fmt = random.choice([
                    f"Qual a temperatura atual em {c} em graus {unit}?",
                    f"Como está o clima em {c} agora? Use {unit}.",
                    f"Verifique as condições meteorológicas de {c} ({unit}).",
                    f"Me informe a previsão do tempo para {c} em {unit}.",
                    f"Consulte o clima em tempo real para a cidade de {c} ({unit})."
                ])
                fn = "get_current_weather"
                args = {"city": c, "unit": unit}
            elif t_type == 1:
                fc = random.choice(CURRENCIES)
                tc = random.choice([c for c in CURRENCIES if c != fc])
                amt = random.randint(10, 25000)
                fmt = random.choice([
                    f"Converta {amt} {fc} para {tc} na cotação de hoje.",
                    f"Quanto dá {amt} {fc} em {tc}?",
                    f"Faça o câmbio de {amt} {fc} para {tc}.",
                    f"Qual o valor correspondente de {amt} {fc} convertidos em {tc}?"
                ])
                fn = "convert_currency"
                args = {"amount": amt, "from_currency": fc, "to_currency": tc}
            elif t_type == 2:
                tz = random.choice(TIMEZONES)
                fmt = random.choice([
                    f"Que horas são agora em {tz}?",
                    f"Informe o horário atual no fuso {tz}.",
                    f"Qual a hora exata no fuso horário {tz}?",
                    f"Consulte o relógio oficial para {tz}."
                ])
                fn = "get_timezone_time"
                args = {"timezone": tz}
            elif t_type == 3:
                u1, u2, fn, vmin, vmax = random.choice(UNIT_PAIRS)
                val = random.randint(vmin, vmax)
                fmt = random.choice([
                    f"Converta {val} {u1} para {u2}.",
                    f"Quanto é {val} {u1} em {u2}?",
                    f"Transforme a medida de {val} {u1} para {u2}."
                ])
                args = {"value": val, "from_unit": u1, "to_unit": u2}
            else:
                topic = random.choice(SEARCH_TOPICS)
                sub = random.choice(["exemplo prático", "melhores práticas", "solução de problemas", "documentação oficial", "guia de início"])
                fmt = random.choice([
                    f"Pesquise na base de conhecimento sobre '{topic}' com foco em {sub}.",
                    f"Consulte na documentação técnica: '{topic}' ({sub}).",
                    f"Busque nos artigos internos sobre '{topic}' - {sub}."
                ])
                fn = "search_knowledge_base"
                args = {"query": f"{topic} {sub}"}

            norm = fmt.lower()
            if norm not in seen:
                seen.add(norm)
                single_samples.append((fmt, fn, args))
                break

    print(f"Generated single-tool samples: {len(single_samples)}")
    assert len(single_samples) == 500

test_l1_generation()
