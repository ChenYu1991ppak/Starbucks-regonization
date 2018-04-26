import os
import json

def parse_record(str):
    return json.loads(str)

def read_recordfile(path):
    """ read record and return dict """
    f = open(path, 'r')
    line = f.readline()
    while line:
        yield parse_record(line)
        line = f.readline()

