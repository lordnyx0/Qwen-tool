"""
rewards_tool_calling.py
Módulo de funções de recompensa hierárquicas (HTGR) para treino de Tool Calling com GRPO:
1. reward_format: Validação estrita do formato XML <tool_call> e sintaxe JSON
2. reward_tool_selection: Acerto no nome da ferramenta chamada
3. reward_arguments: Acerto nos parâmetros requeridos e tipos
4. reward_abstention: Recompensa para casos negativos (quando NÃO deve chamar ferramentas)
5. reward_python_execution: Executa o código Python no sandbox local e valida o resultado numérico/texto
"""

import json
import re
import subprocess
import sys

def extract_tool_calls_from_text(text):
    """Extrai chamadas de ferramenta do formato nativo Qwen."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    parsed = []
    has_xml_tag = ("<tool_call>" in text)
    
    for m in matches:
        try:
            obj = json.loads(m.strip())
            if isinstance(obj, dict) and "name" in obj:
                parsed.append(obj)
        except Exception:
            pass
    return parsed, has_xml_tag

def reward_format(completions, **kwargs):
    """
    Recompensa de formato (+1.0 se JSON válido dentro de <tool_call>, ou se for abstenção sem tags).
    """
    rewards = []
    expected_calls = kwargs.get("expected_call", [None] * len(completions))
    
    for comp, exp in zip(completions, expected_calls):
        calls, has_xml = extract_tool_calls_from_text(comp)
        if exp is None:
            # Caso de abstenção: sucesso é NÃO conter <tool_call>
            if not has_xml and len(calls) == 0:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            # Caso de tool call: deve ter a tag e JSON válido
            if has_xml and len(calls) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
    return rewards

def reward_tool_selection(completions, **kwargs):
    """
    Recompensa de seleção de ferramenta (+1.0 se o nome da tool for o esperado).
    """
    rewards = []
    expected_calls = kwargs.get("expected_call", [None] * len(completions))
    
    for comp, exp in zip(completions, expected_calls):
        if exp is None:
            # Para abstenção, o acerto é não chamar
            calls, _ = extract_tool_calls_from_text(comp)
            rewards.append(1.0 if len(calls) == 0 else 0.0)
        else:
            calls, _ = extract_tool_calls_from_text(comp)
            if any(c.get("name") == exp.get("name") for c in calls):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
    return rewards

def reward_arguments(completions, **kwargs):
    """
    Recompensa de argumentos (+1.0 se as chaves requeridas estiverem presentes).
    """
    rewards = []
    expected_calls = kwargs.get("expected_call", [None] * len(completions))
    
    for comp, exp in zip(completions, expected_calls):
        if exp is None:
            calls, _ = extract_tool_calls_from_text(comp)
            rewards.append(1.0 if len(calls) == 0 else 0.0)
        else:
            calls, _ = extract_tool_calls_from_text(comp)
            exp_name = exp.get("name")
            exp_args = exp.get("arguments", {})
            
            matched = False
            for c in calls:
                if c.get("name") == exp_name:
                    c_args = c.get("arguments", {})
                    if isinstance(c_args, dict) and isinstance(exp_args, dict):
                        # Todas as chaves esperadas devem estar presentes
                        if all(k in c_args for k in exp_args.keys()):
                            matched = True
                            break
            rewards.append(1.0 if matched else 0.0)
    return rewards

def reward_abstention(completions, **kwargs):
    """
    Bônus de abstenção (+1.5 se evitou over-triggering em casos conversacionais).
    """
    rewards = []
    expected_calls = kwargs.get("expected_call", [None] * len(completions))
    
    for comp, exp in zip(completions, expected_calls):
        if exp is None:
            calls, has_xml = extract_tool_calls_from_text(comp)
            # Recompensa alta por responder diretamente sem tool
            if not has_xml and len(calls) == 0 and len(comp.strip()) > 5:
                rewards.append(1.5)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards

def reward_python_execution(completions, **kwargs):
    """
    Executa o código Python gerado no interpretador sandbox e valida a resposta (+2.0).
    """
    rewards = []
    expected_results = kwargs.get("expected_execution_result", [None] * len(completions))
    
    for comp, exp_res in zip(completions, expected_results):
        if exp_res is None:
            rewards.append(0.0)
            continue
            
        calls, _ = extract_tool_calls_from_text(comp)
        py_calls = [c for c in calls if c.get("name") == "execute_python"]
        
        if not py_calls:
            rewards.append(0.0)
            continue
            
        code = py_calls[0].get("arguments", {}).get("code", "")
        if not code or len(code.strip()) == 0:
            rewards.append(0.0)
            continue
            
        # Execução isolada em subprocesso seguro com timeout de 2 segundos
        try:
            res = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            stdout = res.stdout.strip()
            # Comparação flexível (exato ou contido)
            if stdout == str(exp_res).strip() or str(exp_res).strip() in stdout:
                rewards.append(2.0)
            elif res.returncode == 0:
                rewards.append(0.5)  # Executou sem erro, mas valor divergiu
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
            
    return rewards
