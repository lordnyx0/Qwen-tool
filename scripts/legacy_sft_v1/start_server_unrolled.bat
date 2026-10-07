@echo off
setlocal

:: ====================================================================
::  llama-server  --  Qwen3Loop Unrolled (Hotfix 0.3)  --  CUDA GPU 32k
:: ====================================================================

set "LLAMA_DIR=C:\Users\Nyx\llama-cpp"
set "SERVER=%LLAMA_DIR%\llama-server.exe"

:: Modelo unrolled Q8_0 (padrao de alta precisao)
set "MODEL=C:\Users\Nyx\Downloads\Diversos\qwen\unrolled_modelo_qwen3loop_sft_q8_0.gguf"

:: Caso queira alternar para versao F16:
:: set "MODEL=C:\Users\Nyx\Downloads\Diversos\qwen\unrolled_modelo_qwen3loop_sft_f16.gguf"

:: Configuracoes do Servidor Web
set "HOST=0.0.0.0"
set "PORT=8080"
set "CTX=32768"
set "N_GPU_LAYERS=999"
set "N_PARALLEL=4"

:: Parametros calibrados para Qwen3Loop (Hotfix 0.3)
set "TEMP=0.6"
set "TOP_P=0.95"
set "TOP_K=40"
set "REP_PEN=1.08"

echo.
echo  ====================================================================
echo    llama-server ^| Qwen3Loop Unrolled 0.6B (56 layers) ^| CUDA GPU
echo  ====================================================================
echo.
echo  Modelo    : %MODEL%
echo  Host      : %HOST%:%PORT%
echo  Contexto  : %CTX% tokens (32k)
echo  GPU       : %N_GPU_LAYERS% camadas (offload total na GPU)
echo  Slots     : %N_PARALLEL% requisicoes paralelas
echo  Sampling  : Temp=%TEMP% ^| Top-P=%TOP_P% ^| Repeat-Penalty=%REP_PEN%
echo.
echo  Interface Web e Chat:
echo    http://localhost:%PORT%/
echo.
echo  API OpenAI compativel:
echo    http://localhost:%PORT%/v1/chat/completions
echo    http://localhost:%PORT%/v1/models
echo.
echo  --------------------------------------------------------------------
echo  Iniciando servidor...
echo.

"%SERVER%" ^
  --model "%MODEL%" ^
  --host %HOST% ^
  --port %PORT% ^
  --ctx-size %CTX% ^
  --n-gpu-layers %N_GPU_LAYERS% ^
  --parallel %N_PARALLEL% ^
  --batch-size 512 ^
  --ubatch-size 512 ^
  --flash-attn on ^
  --temp %TEMP% ^
  --top-p %TOP_P% ^
  --top-k %TOP_K% ^
  --repeat-penalty %REP_PEN% ^
  --log-prefix

echo.
echo  Servidor encerrado. Pressione qualquer tecla...
pause >nul
