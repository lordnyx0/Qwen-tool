@echo off
setlocal
REM =====================================================================
REM  Qwen3Loop Tool V3 — llama-server para o Pi coding agent (GPU)
REM  Uso: duplo-clique. Depois, no Pi: /login llama.cpp (http://127.0.0.1:8080)
REM =====================================================================

REM --- Ajuste aqui o caminho do binario (winget install llama.cpp) ---
set "LLAMA_SERVER=C:\Users\Nyx\llama-cpp\llama-server.exe"
set "MODELS_DIR=C:\Users\Nyx\Downloads\Diversos\qwen\models\gguf"
set "PORT=8080"
set "CTX=16384"

if not exist "%LLAMA_SERVER%" (
  echo [ERRO] llama-server.exe nao encontrado em:
  echo   %LLAMA_SERVER%
  echo.
  echo Instale com:  winget install llama.cpp   ^(build com CUDA^)
  echo Ou baixe o release CUDA em https://github.com/ggml-org/llama.cpp/releases
  echo e ajuste a variavel LLAMA_SERVER no topo deste .bat
  pause
  exit /b 1
)
if not exist "%MODELS_DIR%" (
  echo [ERRO] Diretorio de modelos nao encontrado em:
  echo   %MODELS_DIR%
  pause
  exit /b 1
)

echo [OK] Router llama.cpp na porta %PORT% servindo %MODELS_DIR% ...
echo No Pi: /login llama.cpp -^> /llama ^(carregar modelo^) -^> /model ^(selecionar^).
echo Pressione Ctrl+C para encerrar.
echo.

REM --- Router mode (exigido pelo Pi): sem -m, com --models-dir.
REM --- --jinja (tool calling) / -ngl 999 (tudo na GPU) / -c 16384
REM --- (KV q8_0 ~= 1.8 GB + modelo 1.0 GB, folga na RTX 3060 12GB)
"%LLAMA_SERVER%" --models-dir "%MODELS_DIR%" --no-models-autoload ^
  --host 127.0.0.1 --port %PORT% ^
  --jinja -ngl 999 -c %CTX% ^
  --cache-type-k q8_0 --cache-type-v q8_0 ^
  --flash-attn on

REM Se o servidor fechar com erro de flag desconhecida, remova a linha
REM --flash-attn acima (build sem suporte a Flash Attention no Ampere).

echo.
echo [INFO] Servidor encerrado.
pause
