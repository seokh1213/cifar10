# Cifar 10 with several appraoch.

## Same settings.
```python
config = {
        'batch_size': 128,
        'base_lr': 1e-3,
        'num_epochs': 100,
        'valid_ratio': 0.1,
        'seed': 1213,
    }
optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=20,
                                          cycle_mult=1.0,
                                          max_lr=1e-3,
                                          min_lr=1e-8,
                                          warmup_steps=0,
                                          gamma=0.8)
criterion = nn.CrossEntropyLoss()
```

## Model
> ### Shallow Model
> ****  
>    
> ### Using Resnet Structure Model
> ****  
>  
>### Using Efficientnet Structure
> ****  
>
> ### Pretrained Model
> * EfficientNet 0
> * Resnet 18
  
## Approach
> ### Basic
> ****  
> ### Self Supervised Learning
> ****  
> ### Using AutoEncoder Features
> ****  

## Visualization CAM
