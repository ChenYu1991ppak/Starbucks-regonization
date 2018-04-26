import os
from PIL import Image
import json

from torch.utils.data import Dataset

from utils import read_recordfile
from preprocess import image2matrix, get_chunks_center, clustering, flattening_with_weight, cut_matrix, matrix2image

samples_dir = os.path.join("xingbake", "api")


# def import_samples():
#     for record in read_recordfile(record_path):
#         sample = Image.open(os.path.join(samples_dir, record["name"]))
#         label = record["target"]
#         yield sample, label

def read_sample(record):
    sample = Image.open(os.path.join(samples_dir, record["name"]))
    label = record["target"]
    return sample, label

def normalize_image(img):
    img = img.transpose(2, 0, 1)
    img = img / 255
    return img

def make_index(label):
    if label == "1":
        label = "I"
    if label == "6":
        label = "B"
    if label == "8":
        label = "B"
    if label == "0":
        label = "O"

    assert ord(label) >= ord("A") and ord(label) <= ord("Z"), "range error"
    return ord(label) - ord("A")

def make_count(label):
    l = len(label)
    if l <= 4:
        return 0
    elif l == 5:
        return 1
    elif l >= 6:
        return 2



def read_chunks(path):
    for j, record in enumerate(read_recordfile(path)):
        if j % 10 == 0:
            print("%d pictures has been read" % j)
        image = Image.open(os.path.join(samples_dir, record["name"]))
        label = record["target"]
        cls = len(label)
        weight_in_col = flattening_with_weight(image)
        cls_list = clustering(weight_in_col, cluster=cls)

        chunks_center = get_chunks_center(image, cls_list)
        for i in range(len(record["target"])):
            yield record["name"], chunks_center[i], label[i]

def read_image_w_label(path):
    for j, record in enumerate(read_recordfile(path)):
        image = record["name"]
        label = record["target"]
        yield image, label


class chunksDataset(Dataset):
    def __init__(self, record, dir=samples_dir):
        self.record = record
        self.dir = dir

    def __getitem__(self, index):
        re = self.record[index]
        image = Image.open(os.path.join(self.dir, re[0]))
        center = re[1]
        matrix = image2matrix(image)
        matrix_new = cut_matrix(matrix, center[0], center[1])
        matrix_new = normalize_image(matrix_new)
        label = re[2]
        label = make_index(label)
        return matrix_new, label

    def __len__(self):
        return len(self.record)

class imgDataset(Dataset):
    def __init__(self, record, dir=samples_dir):
        self.record = record
        self.dir = dir

    def __getitem__(self, index):
        re = self.record[index]
        image = Image.open(os.path.join(self.dir, re[0]))
        label = re[1]
        image = image2matrix(image)
        image = normalize_image(image)
        label = make_count(label)
        return image, label

    def __len__(self):
        return len(self.record)


# if __name__ == "__main__":
#     a = import_samples()
#     print(next(a)[0].show())
