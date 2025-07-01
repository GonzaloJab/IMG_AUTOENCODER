# IMG_AUTOENCODER
Simple implementation of encoder-decoder architecture for images:

- Image size: 256 × 2048
- Grayscale images
- Autoencoder for anomaly detection

## Training

### Basic Training
```bash
python train.py --data_dir /path/to/your/images --epochs 20
```

### Resume Training from Checkpoint
```bash
python train.py --data_dir /path/to/your/images --epochs 10 --resume_from ./checkpoints/autoencoder_epoch_10.pth
```

### List Available Checkpoints
```bash
python list_checkpoints.py
```

## Usage Examples

1. **Start new training:**
   ```bash
   python train.py --data_dir E:\0_DATASETS\NONE --epochs 20 --batch_size 2
   ```

2. **Continue training from epoch 10:**
   ```bash
   python train.py --data_dir E:\0_DATASETS\NONE --epochs 10 --resume_from ./checkpoints/autoencoder_epoch_10.pth
   ```

3. **Check available checkpoints:**
   ```bash
   python list_checkpoints.py
   ```

## Features

- ✅ Resume training from any checkpoint
- ✅ Automatic epoch numbering when resuming
- ✅ Optimizer state preservation
- ✅ Backward compatibility with old checkpoint format
- ✅ Progress bars and detailed logging


