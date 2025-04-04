# E-commerce Review Sentiment Classifier

A machine learning project to classify customer sentiments from e-commerce product reviews into **Negative**, **Positive**, or **Neutral**. Built to help businesses understand customer feedback at scale—perfect for review analysis, customer support, or product improvement.

## Overview
- **Model**: Fine-tuned [DistilBERT](https://huggingface.co/distilbert-base-uncased) for sentiment classification.
- **Dataset**: 10,000 Amazon product reviews, mapped from 1-5 star ratings to sentiment labels (1-2: Negative, 3: Neutral, 4-5: Positive).
- **Performance**: Achieved **87.1% validation accuracy** after 3 epochs.
- **Tech Stack**: Python, Hugging Face Transformers, PyTorch, Google Colab (free GPU).
- **Use Case**: Real-time sentiment analysis for e-commerce platforms, customer insights, or automated review moderation.

## Files
- **`sentiment_classifier.py`**: Complete code for training and inference. Includes data preprocessing, model fine-tuning, and sentiment prediction.
- **`sentiment_model.zip`**: Trained DistilBERT model (unzip to `./sentiment_model/final` for inference).

## How It Works
1. **Training**: Fine-tuned DistilBERT on 20K reviews with a custom dataset (text + scores).
2. **Inference**: Predicts sentiment from any review text with confidence scores.
3. **Deployment**: Lightweight and optimized for free-tier GPU environments like Google Colab.

## Sample Predictions
| Review Text                              | Predicted Sentiment | Confidence |
|------------------------------------------|---------------------|------------|
| "This product is amazing, worth it!"     | Positive            | 0.95       |
| "Terrible quality, broke in two days."  | Negative            | 0.89       |
| "It’s okay, nothing special."           | Neutral             | 0.78       |

## Setup and Usage
### Prerequisites
- Python 3.8+
- Libraries: `transformers`, `torch`, `datasets`
- GPU recommended (runs on Colab/Kaggle free tiers)

### Steps
1. **Clone the repo**:
   ```bash
   git clone https://github.com/hmankar01/ecommerce-sentiment-classifier.git
   cd ecommerce-sentiment-classifier
