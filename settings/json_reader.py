import json
def read_json(file):
    file = dict(json.load(open(file)))
    file = file