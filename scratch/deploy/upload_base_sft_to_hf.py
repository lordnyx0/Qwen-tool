"""
upload_base_sft_to_hf.py
Cria o repositório privado no Hugging Face e faz upload do dataset do primeiro treino (Base SFT).
"""

import os
from huggingface_hub import HfApi, create_repo

api = HfApi()

REPO_ID = "Lordnyx/qwen3loop-base-sft-dataset"
REPO_TYPE = "dataset"

print(f"1. Criando repositório privado: {REPO_ID} (tipo: {REPO_TYPE})...")
try:
    repo_url = create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=True,
        exist_ok=True
    )
    print(f"Repositório pronto: {repo_url}")
except Exception as e:
    print(f"Aviso ao criar/verificar repo: {e}")

# 2. Criar README.md (Dataset Card Completo)
README_CONTENT = """---
license: apache-2.0
language:
- pt
- en
tags:
- qwen
- depth-recurrence
- layer-looping
- sft
- multi-exit
- halting-probe
- laya-curation
size_categories:
- 1K<n<10K
task_categories:
- text-generation
- question-answering
---

# Qwen3Loop-0.6B: Base SFT Dataset (Primeiro Treinamento)

Este repositório privado armazena o dataset supervisionado (**Base SFT**) de 6.324 amostras curadas e anotadas, utilizado no primeiro treinamento do modelo **Qwen3Loop-0.6B** (Depth-Recurrent Transformer com 594M parâmetros físicos e 56 camadas lógicas unrolled).

---

## 📌 O que é este Dataset?

Este dataset foi construído e curado para fundar as capacidades analíticas densas (*Chain-of-Thought*) e o controle de recorrência em camadas do Qwen3Loop antes da especialização agêntica. 

Diferente de um dataset padrão de conversação, cada amostra foi selecionada pelo pipeline de curadoria **Laya**, avaliada em qualidade e anotada com metadados de **dificuldade e índice de recorrência** para alimentar conjuntamente:
1. A **Multi-Exit Intermediate Loss** ($w_1 = 0.4, w_2 = 1.0$).
2. A cabeça de classificação latente **Latent Halting Probe Head** ($1024 \to 64 \to 3$).

---

## 🔬 Especificações do Dataset

| Característica | Valor |
| :--- | :--- |
| **Total de Amostras** | 6.324 exemplos |
| **Tamanho do Arquivo** | ~25.3 MB (`sft_annotated_curated.jsonl`) |
| **Formato de Tokenização** | Qwen ChatML (`<|im_start|>system...`, `<|im_start|>user...`, `<|im_start|>assistant...`) |
| **Campos por Amostra** | `id`, `source`, `task`, `quality_score`, `messages`, `meta` |
| **Domínios Cobertos** | Raciocínio Matemático (Math CoT), Lógica Algorítmica, Raciocínio Simbólico, Resolução de Problemas Complexos |

---

## 📑 Estrutura de Cada Amostra (`JSONL`)

```json
{
  "id": "eurus_3533",
  "source": "eurus-2-agentic",
  "task": "Math",
  "quality_score": 153.05,
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "meta": {
    "difficulty": "hard",
    "needs_recurrence": true,
    "recurrence_value": 3,
    "early_exit": false
  }
}
```

### Detalhamento dos Campos:
- **`quality_score`:** Pontuação contínua calculada pelo framework de curadoria **Laya**, avaliando a solidez dos passos analíticos e a ausência de alucinações.
- **`meta.difficulty`:** Nível de complexidade inferido (`easy`, `medium`, `hard`).
- **`meta.needs_recurrence`:** Booleano indicando se o problema exige refinamento iterativo profundo (Loop 2) ou se pode ser resolvido com uma única passagem rápida (Loop 1 / Early Exit).
- **`meta.recurrence_value`:** Rótulo alvo para a cabeça de classificação da **Halting Probe** (0 = Saída antecipada, 1 = Moderado, 2 = Recorrência profunda).

---

## ⚙️ Metodologia de Treino que Utilizou este Dataset

- **Arquitetura:** `Qwen3LoopForCausalLM` (28 camadas físicas, 2 loops autorregressivos consecutivos).
- **Otimizador:** BitsAndBytes AdamW 8-bit ($\text{LR} = 2 \times 10^{-5}$, cosine decay).
- **Função de Perda:**
  $$\mathcal{L}_{\text{total}} = 0.4 \cdot \mathcal{L}_{\text{LM}}^{(1)} + 1.0 \cdot \mathcal{L}_{\text{LM}}^{(2)} + 0.15 \cdot \mathcal{L}_{\text{probe}}$$
- **Resultado do Treino:** Checkpoint base FP16 convergido com LM Loss ~0.28 e acurácia da sonda de halting >84%.

---

## 📂 Arquivos Incluídos

- `sft_annotated_curated.jsonl` (25.3 MB): Dataset mestre 100% anotado com pontuação Laya e metadados de recorrência.
- `README.md`: Este guia de documentação e contexto técnico para arquivamento e reprodução futura.
"""

# Salvar README temporário para upload
with open("scratch/HF_DATASET_README.md", "w", encoding="utf-8") as f:
    f.write(README_CONTENT)

print("2. Enviando README.md...")
api.upload_file(
    path_or_fileobj="scratch/HF_DATASET_README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)
print("README.md enviado com sucesso!")

print("3. Enviando sft_annotated_curated.jsonl (25.3 MB)...")
api.upload_file(
    path_or_fileobj="data/sft_annotated_curated.jsonl",
    path_in_repo="sft_annotated_curated.jsonl",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)
print("sft_annotated_curated.jsonl enviado com sucesso!")

print(f"\n✅ Upload concluído com sucesso!")
print(f"Repositório Privado no Hugging Face: https://huggingface.co/datasets/{REPO_ID}")
