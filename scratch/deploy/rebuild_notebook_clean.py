import json

nb_path = 'kaggle_kernel/qwen3loop_tool_sft.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Rebuild cells cleanly
cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Qwen3Loop-0.6B: Full-Parameter Tool-Calling SFT\n",
            "### Fine-Tuning Supervisionado Recorrente com Multi-Exit Loss e AdamW 8-bit\n",
            "\n",
            "> **Status:** Treinamento supervisionado dos 594M parâmetros físicos com pesos amarrados (tied loop), unrolling 56 camadas lógicas e garantia de sobrevivência à quantização GGUF Q8_0."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Verificar GPU Tesla T4 ativa\n",
            "!nvidia-smi"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Instalar dependências necessárias\n",
            "!pip uninstall -y torchao\n",
            "!pip install -q transformers datasets accelerate bitsandbytes"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Mapear e preparar diretório de trabalho\n",
            "import os, sys, shutil, tarfile\n",
            "\n",
            "print('Configurando ambiente no Kaggle...')\n",
            "\n",
            "# Descompactar arquivos .tar se presentes em /kaggle/input\n",
            "for root, dirs, files in os.walk('/kaggle/input'):\n",
            "    for f in files:\n",
            "        if f.endswith('.tar'):\n",
            "            tar_p = os.path.join(root, f)\n",
            "            print('Descompactando tar:', tar_p)\n",
            "            try:\n",
            "                with tarfile.open(tar_p) as t:\n",
            "                    t.extractall('/kaggle/working')\n",
            "            except Exception as e:\n",
            "                print('Aviso ao extrair:', e)\n",
            "\n",
            "# 1. Localizar engine qwen3loop_python: priorizar a versão atualizada\n",
            "engine_src = None\n",
            "search_paths = ['/kaggle/working', '/kaggle/input']\n",
            "for base in search_paths:\n",
            "    for root, dirs, files in os.walk(base):\n",
            "        if 'modeling_qwen3loop.py' in files and 'configuration_qwen3loop.py' in files:\n",
            "            if 'sft-data' in root or '/kaggle/working' in root:\n",
            "                engine_src = root\n",
            "                break\n",
            "            elif engine_src is None:\n",
            "                engine_src = root\n",
            "    if engine_src and ('/kaggle/working' in engine_src or 'sft-data' in engine_src):\n",
            "        break\n",
            "\n",
            "print('engine_src selecionada:', engine_src)\n",
            "assert engine_src is not None, 'Engine qwen3loop_python não encontrada!'\n",
            "\n",
            "# Copiar engine para /kaggle/working/qwen3loop_python e /kaggle/working/engine/qwen3loop_python\n",
            "os.makedirs('/kaggle/working/engine', exist_ok=True)\n",
            "if engine_src != '/kaggle/working/qwen3loop_python':\n",
            "    shutil.copytree(engine_src, '/kaggle/working/qwen3loop_python', dirs_exist_ok=True)\n",
            "if engine_src != '/kaggle/working/engine/qwen3loop_python':\n",
            "    shutil.copytree(engine_src, '/kaggle/working/engine/qwen3loop_python', dirs_exist_ok=True)\n",
            "\n",
            "# Inserir no início absoluto de sys.path e PYTHONPATH\n",
            "sys.path.insert(0, '/kaggle/working')\n",
            "sys.path.insert(0, '/kaggle/working/engine')\n",
            "os.environ['PYTHONPATH'] = f'/kaggle/working:/kaggle/working/engine:{os.environ.get(\"PYTHONPATH\", \"\")}'\n",
            "\n",
            "# Testar importação e registro formal da arquitetura\n",
            "import qwen3loop_python\n",
            "from qwen3loop_python import Qwen3LoopConfig, Qwen3LoopForCausalLM\n",
            "from transformers import AutoConfig, AutoModelForCausalLM\n",
            "AutoConfig.register('qwen3loop', Qwen3LoopConfig)\n",
            "AutoModelForCausalLM.register(Qwen3LoopConfig, Qwen3LoopForCausalLM)\n",
            "print('Engine qwen3loop_python importada e registrada com sucesso de:', getattr(qwen3loop_python, '__file__', 'unknown'))\n",
            "\n",
            "# 2. Localizar Modelo Base e Dados\n",
            "model_path = None\n",
            "for root, dirs, files in os.walk('/kaggle/input'):\n",
            "    if 'model.safetensors' in files and 'config.json' in files:\n",
            "        model_path = root\n",
            "        break\n",
            "\n",
            "data_path = None\n",
            "for base in ['/kaggle/working', '/kaggle/input']:\n",
            "    for root, dirs, files in os.walk(base):\n",
            "        if 'sft_ultradata_tool_use_8k.jsonl' in files:\n",
            "            data_path = os.path.join(root, 'sft_ultradata_tool_use_8k.jsonl')\n",
            "            break\n",
            "    if data_path:\n",
            "        break\n",
            "\n",
            "script_path = None\n",
            "for base in ['/kaggle/working', '/kaggle/input']:\n",
            "    for root, dirs, files in os.walk(base):\n",
            "        if 'train_sft_tool_qwen3loop.py' in files:\n",
            "            script_path = os.path.join(root, 'train_sft_tool_qwen3loop.py')\n",
            "            break\n",
            "    if script_path:\n",
            "        break\n",
            "\n",
            "print('Modelo Base detectado:', model_path)\n",
            "print('Dataset SFT detectado:', data_path)\n",
            "print('Script SFT detectado: ', script_path)\n",
            "assert model_path is not None, 'Modelo não encontrado!'\n",
            "assert data_path is not None, 'Dataset não encontrado!'\n",
            "assert script_path is not None, 'Script de treino não encontrado!'\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Executar Treinamento Full-Parameter SFT 8K em Dual GPU Tesla T4 (Accelerate DDP)\n",
            "output_dir = '/kaggle/working/qwen3loop_tool_sft_final'\n",
            "\n",
            "!accelerate launch --multi_gpu --num_processes 2 {script_path} \\\n",
            "    --model_path {model_path} \\\n",
            "    --data_path {data_path} \\\n",
            "    --output_dir {output_dir} \\\n",
            "    --batch_size 1 \\\n",
            "    --grad_accum 8 \\\n",
            "    --epochs 2 \\\n",
            "    --lr 1.5e-5 \\\n",
            "    --max_len 8192 \\\n",
            "    --eval_steps 30 \\\n",
            "    --save_steps 90\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Compactar os pesos finais do modelo para download direto\n",
            "!cd /kaggle/working && zip -r qwen3loop_tool_sft_final.zip qwen3loop_tool_sft_final/\n",
            "!ls -lh /kaggle/working/qwen3loop_tool_sft_final.zip\n",
            "print('\\nTreinamento SFT concluído com sucesso! Checkpoint zip gerado para download e unrolling.')"
        ]
    }
]

nb['cells'] = cells

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Rebuilt kaggle notebook with all 6 cells cleanly!')
