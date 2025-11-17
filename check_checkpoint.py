#!/usr/bin/env python3
import torch

ckpt = torch.load('/cs/student/projects1/2023/muhamaaz/checkpoints/model1_custom_v2_fixed_wikiart/best_model.pt', 
                  map_location='cpu', weights_only=False)

print('Checkpoint Info:')
print(f"  Epoch: {ckpt.get('epoch')}")
print(f"  Val Loss: {ckpt.get('best_val_loss')}")
print(f"  Training Info: {ckpt.get('training_info', 'N/A')}")
