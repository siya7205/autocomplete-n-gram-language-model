import argparse

from autocomplete.sentiment import predict_sentiment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict sentiment for a single text input.")
    parser.add_argument("--text", required=True, help="Input text to classify.")
    parser.add_argument("--model", default="models/sentiment.pkl", help="Path to trained sentiment model.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    label, scores = predict_sentiment(text=args.text, model_path=args.model)

    print(f"Text: {args.text}")
    print(f"Predicted label: {label}")
    if scores:
        print("Class probabilities:")
        for class_label, value in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            print(f"- {class_label}: {value:.6f}")


if __name__ == "__main__":
    main()
