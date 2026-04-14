import argparse

from autocomplete.sentiment import train_sentiment_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline sentiment classifier from labeled CSV.")
    parser.add_argument("--csv", required=True, help="Path to labeled CSV (id,text,sentiment_label,notes).")
    parser.add_argument("--out", default="models/sentiment.pkl", help="Path to save trained model artifact.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible train/test split.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    metrics = train_sentiment_model(csv_path=args.csv, model_out_path=args.out, seed=args.seed)

    print(f"Model saved: {metrics['model_path']}")
    print(f"Labeled rows: {metrics['labeled_rows']}")
    print(f"Train rows: {metrics['train_rows']}")
    print(f"Test rows: {metrics['test_rows']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"Labels: {', '.join(metrics['labels'])}")
    print("Confusion matrix (rows=true, cols=pred):")
    for row in metrics["confusion_matrix"]:
        print(" ".join(str(value) for value in row))


if __name__ == "__main__":
    main()
