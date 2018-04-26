import os

import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np

from load_dataset import normalize_image, image2matrix
from CNN import CNN
from CNN2 import CNN2
from preprocess import flattening_with_weight, clustering, get_chunks_center, cut_matrix


model_path = os.path.join(os.getcwd(), "models")
model_path2 = os.path.join(os.getcwd(), "models2")
model = CNN(num_classes=26).cpu()
model.load_state_dict(torch.load(os.path.join(model_path, "model43-1.pkl")))
model2 = CNN2(num_classes=3)
model2.load_state_dict(torch.load(os.path.join(model_path2, "model67-1.pkl")))


test_path = os.path.join(os.getcwd(), "test")

dtype1 = torch.FloatTensor
dtype2 = torch.LongTensor

# def label2location(label):
#     location = label * pix_size
#     return location


def image2chunks(image):
    matrix = image2matrix(image)
    data = normalize_image(matrix)
    input = torch.from_numpy(data).type(dtype1)
    c, h, w = input.size()
    input = Variable(input.expand(1, c, h, w))
    output = model2(input)
    _, predict = torch.max(output.data, 1)
    predict = int(predict.cpu().numpy())
    cls = predict + 4

    weight_in_col = flattening_with_weight(image)
    cls_list = clustering(weight_in_col, cluster=cls)
    chunks_center = get_chunks_center(image, cls_list)
    return chunks_center


def predict(img):
    chunks_center = image2chunks(img)
    matrix = image2matrix(img)
    result = ""
    for c in chunks_center:
        sub_matrix = cut_matrix(matrix, c[0], c[1])
        data = normalize_image(sub_matrix)
        input = torch.from_numpy(data).type(dtype1)
        c, h, w = input.size()
        input = Variable(input.expand(1, c, h, w))
        output = model(input)
        _, predict = torch.max(output.data, 1)
        predict = int(predict.cpu().numpy())
        re = chr(predict + ord("A"))
        result += re

    return result


# if __name__ == "__main__":
    # files = list(find_images(test_path))
    # print(len(files))
    # for f in files:
    #     img = Image.open(os.path.join(test_path, f))
    #     re = predict(img)
    #     img.show()
    #     print(re, f)
