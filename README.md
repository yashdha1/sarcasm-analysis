# Sarcasm Classification with BERT

## Overview
This project uses BERT to classify tweets into four categories:
- Sarcasm
- Irony
- Figurative
- Regular

The model is fine-tuned on a labeled dataset and achieves high accuracy in detecting sarcastic and related tones.

## Requirements
- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- Pandas
- NLTK
- Tqdm

Install dependencies using:
```bash
pip install torch transformers pandas nltk tqdm
```

## How to Run
1. Prepare your dataset files:
   - `train.csv`: Training data with `tweets` and `class` columns.
   - `test.csv`: Testing data with the same structure.

2. Run the script to train and evaluate the model:
```bash
python sarcasm_classification.py
```

3. The best model will be saved as `best_model.pt`.
