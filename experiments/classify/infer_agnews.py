from transformers import pipeline
clf = pipeline("text-classification", model="runs/day22/final")
print(clf("Phelps wins gold in the 200 meter freestyle"))
print(clf("Wall Street shares fall after the rate decision"))