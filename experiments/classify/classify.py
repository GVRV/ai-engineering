import sys
from transformers import pipeline

def main():
    text = " ".join(sys.argv[1:]).strip()

    if not text:
        print("usage: python classify.py <headline>")
        return

    clf = pipeline(
        "text-classification",
        model="GVRV/ag_news",
    )
    print(clf(text))

if __name__ == "__main__":
    main()