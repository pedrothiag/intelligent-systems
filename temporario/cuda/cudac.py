import torch

print(f"PyTorch version: {torch.__version__}")

# Verifica se a GPU está disponível
print(f"CUDA Available: {torch.cuda.is_available()}")

# Verifica a versão do CUDA que o PyTorch está usando
print(f"CUDA Version: {torch.version.cuda}")

# Nome da GPU
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
