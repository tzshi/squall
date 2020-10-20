# -*- coding: utf-8 -*-

import json

with open("../data/squall.json") as f:
    squall_data = json.load(f)

dev_ids = []

for i in range(5):
    with open("../data/dev-{}.ids".format(i)) as f:
        dev_ids.append(set(json.load(f)))

for i in range(5):
    dev_set = [x for x in squall_data if x["tbl"] in dev_ids[i]]
    train_set = [x for x in squall_data if x["tbl"] not in dev_ids[i]]

    with open("../data/dev-{}.json".format(i), "w") as f:
        json.dump(dev_set, f, indent=2)

    with open("../data/train-{}.json".format(i), "w") as f:
        json.dump(train_set, f, indent=2)
