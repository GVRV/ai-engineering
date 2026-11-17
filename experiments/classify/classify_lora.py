import sys
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

def main():
    text = " ".join(sys.argv[1:]).strip()

    if not text:
        print("usage: python classify_lora.py <headline>")
        return


    id2label = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Sci/Tech",
    }
    label2id = {v: k for k, v in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained("./runs/day37/lora")
    base = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=4,
        id2label=id2label,
        label2id=label2id,
    )
    model = PeftModel.from_pretrained(base, "./runs/day37/lora")
    model = model.merge_and_unload()

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
    )
    print(clf(text))

if __name__ == "__main__":
    main()