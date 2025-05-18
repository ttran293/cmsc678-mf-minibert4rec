# Movie Recommendation System

This project implements a movie recommendation system using two different approaches:
1. A BERT-style transformer model for sequential movie recommendations
2. A hybrid model combining Matrix Factorization (MF) with the transformer model

## Features

- Sequential movie recommendation using transformer architecture
- Matrix Factorization for collaborative filtering
- Hybrid model combining both approaches
- Evaluation using Hit@K and NDCG@K metrics
- Training visualization and logging
- Support for both CPU and GPU training

## Requirements

- Python 3.7+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- tqdm

### Installation

You can install all required packages using pip and the provided requirements.txt:

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install torch pandas numpy scikit-learn matplotlib tqdm
```

## Data

The project uses the MovieLens dataset. By default, it uses the small version (`ml-latest-small`), but you can also use the full dataset (`ml-latest`). The dataset should contain:
- `ratings.csv`: User ratings data
- `movies.csv`: Movie metadata

## Usage

### Basic Usage

```bash
python main.py
```

This will run both models (BERT and Hybrid) with default parameters.

### Command Line Arguments

- `--model`: Choose which model to run
  - `bert`: Only run the BERT model
  - `hybrid`: Only run the hybrid model
  - `both`: Run both models (default)

- `--epochs`: Number of training epochs for BERT/hybrid models
  - Default: 20 for BERT, 10 for hybrid

- `--user`: Specify a user ID to get recommendations for
  - If not specified, uses a random user from the test set

- `--skip-train`: Skip training and load existing models
  - Useful for just getting recommendations

- `--device`: Choose device for training
  - `cpu` (default)
  - `cuda` for GPU training

### MF/Hybrid Model Parameters

- `--mf-epochs`: Number of epochs for MF pretraining
  - Default: 5

- `--mf-batch-size`: Batch size for MF training
  - Default: 1024

- `--mf-lr`: Learning rate for MF training
  - Default: 0.001

- `--freeze-ratio`: Fraction of hybrid epochs to freeze MF embeddings
  - Default: 0.3

- `--data-dir`: Directory containing the dataset
  - Default: 'ml-latest-small'

### Examples

1. Train and evaluate both models:
```bash
python main.py
```

2. Train only the BERT model with 30 epochs:
```bash
python main.py --model bert --epochs 30
```

3. Get recommendations for a specific user using pre-trained models:
```bash
python main.py --user 123 --skip-train
```

4. Train hybrid model with custom MF parameters:
```bash
python main.py --model hybrid --mf-epochs 10 --mf-batch-size 2048 --mf-lr 0.0005
```

5. Use GPU for training:
```bash
python main.py --device cuda
```

## Output

The script will:
1. Train the selected model(s)
2. Save model checkpoints in the `model/exp_[timestamp]` directory
3. Generate training plots and logs
4. Print recommendations for the selected user

For each user, it shows:
- Their movie watching history
- The held-out next movie (for evaluation)
- Top-5 movie recommendations from each model

## Model Architecture

### BERT Model
- Transformer-based architecture
- Uses movie sequences to predict next movies
- Default configuration:
  - 64-dimensional embeddings
  - 4 attention heads
  - 2 transformer layers

### Hybrid Model
- Combines Matrix Factorization with BERT
- Uses MF embeddings to initialize the transformer
- Two-phase training:
  1. Freeze MF embeddings
  2. Fine-tune all parameters

## Evaluation

Models are evaluated using:
- Hit@K: Whether the true next movie is in top-K recommendations
- NDCG@K: Normalized Discounted Cumulative Gain at K

Results are logged in CSV files in the experiment directory. 