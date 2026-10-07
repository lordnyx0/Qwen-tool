"""Scratch: sobe v38 Q8 (tool) na 8084 p/ coleta best-of-n."""
import subprocess

tpl = open("C:/Users/Nyx/Downloads/Diversos/qwen/Kaggle/kaggle_kernel_output/"
           "sft_v38/qwen3loop_tool_curriculum_3000_final/epoch-2/chat_template.jinja",
           encoding="utf-8").read()
print("template chars:", len(tpl))
subprocess.Popen(
    ["C:/Users/Nyx/llama-cpp/llama-server.exe",
     "-m", "C:/Users/Nyx/Downloads/Diversos/qwen/models/gguf/"
            "unrolled_modelo_qwen3loop_tool_sft_v38_ep2_q8_0.gguf",
     "--chat-template", tpl,
     "--port", "8084", "-c", "8192", "-ngl", "99", "--temp", "0.7"],
    stdout=None, stderr=None, stdin=None,
    creationflags=subprocess.CREATE_NEW_CONSOLE
    | subprocess.CREATE_NEW_PROCESS_GROUP)
print("v38-tool na 8084")
