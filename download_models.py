import torch
import os

# Set cache directory
os.environ['TORCH_HOME'] = '/opt/render/project/src/models'

# Create models directory
if not os.path.exists('/opt/render/project/src/models'):
    os.makedirs('/opt/render/project/src/models')

print("Downloading MiDaS model...")
try:
    # Pre-download the model during build
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    print("Model downloaded successfully!")
except Exception as e:
    print(f"Failed to download model: {e}")
