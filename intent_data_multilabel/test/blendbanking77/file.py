import json

dataset = {}
with open("blendbanking77_val.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    for raw in data["data"]:
        for la in raw["intent"]:
            if la not in data:
                dataset[la] = 1
            else:
                dataset[la] += 1
print(dataset)
