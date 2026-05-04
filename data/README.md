# Dataset Directory

## RML2016.10a

This directory should contain the `RML2016.10a_dict.pkl` dataset file.

**This file is NOT tracked by git** (too large, ~500MB).

### How to get the dataset

1. **Download** from [RadioML](https://www.deepsig.ai/datasets/) or use the shared Google Drive link provided in the team chat.
2. **Place** the file here: `data/RML2016.10a_dict.pkl`
3. **For Google Colab**: Upload to your Google Drive at `MyDrive/AMR-Project/RML2016.10a_dict.pkl`

### Verification

After placing the file, run:
```python
import pickle
Xd = pickle.load(open('data/RML2016.10a_dict.pkl', 'rb'), encoding='iso-8859-1')
print(f"Keys: {len(Xd.keys())}")        # Should be 220
print(f"Sample shape: {list(Xd.values())[0].shape}")  # Should be (1000, 2, 128)
```

Expected output:
```
Keys: 220
Sample shape: (1000, 2, 128)
```
