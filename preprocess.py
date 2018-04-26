from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


chunk_size = 30

def image2matrix(image):
    """A helper for ImageConverted"""
    return np.asarray(image)

def matrix2image(matrix):
    """A helper for ImageConverted"""
    return Image.fromarray(matrix)

def to_binary_image(image):
    image = image.convert("L")
    threshold = 210
    table = [0 if i < threshold else 1 for i in range(256)]
    return image.point(table, "1")

def count_head_rear(sub_matrix):
    head = -1;
    rear = -1
    for i in range(len(sub_matrix)):
        if sub_matrix[i] == 0:
            head = i
            break
    for j in range(len(sub_matrix) - 1, -1, -1):
        if sub_matrix[j] == 0:
            rear = j
            break
    return head, rear

def flattening_matrix(matrix):
    weight_in_col = []
    for col in range(matrix.shape[1]):
        sub_matrix = matrix[:, col]
        head, rear = count_head_rear(sub_matrix)
        weight_in_col.append(abs(head - rear))
    return weight_in_col

def flattening_with_weight(image):
    image = to_binary_image(image)
    matrix = image2matrix(image)
    return flattening_matrix(matrix)

def clustering(weight_in_col, cluster=3):
    pixs = []
    for i, c in enumerate(weight_in_col):
        w = [i for _ in range(c)]
        pixs.extend(w)
    pixs = [[i, 0] for i in pixs]
    pixs = np.array(pixs).reshape(-1, 2)
    kmeans_model = KMeans(n_clusters=cluster).fit(pixs)

    result = []
    for i, c in enumerate(kmeans_model.labels_):
        pix_re = [pixs[i][0], c]
        if pix_re not in result:
            result.append(pix_re)

    cls = result[0][1]
    temp = []
    cls_list = []       # finial result
    for s in result:
        if s[1] == cls:
            temp.append(s[0])
        else:
            cls_list.append(temp)
            temp = []
            cls = s[1]
            temp.append(s[0])
    cls_list.append(temp)
    return cls_list

def get_chunks_center(image, cls_list):
    image_b = to_binary_image(image)
    matrix_b = image2matrix(image_b)
    chunks_center = []
    for i, c in enumerate(cls_list):
        mid_h = int((min(c) + max(c) + 1) / 2)
        head, rear = count_head_rear(matrix_b[:, mid_h])
        mid_v = int((head + rear + 1) / 2)
        chunks_center.append((mid_h, mid_v))
    return chunks_center



def cut_matrix(matrix, x, y):
    top = y - int((chunk_size - 1) / 2)
    bottom = y + int(chunk_size / 2)
    left = x - int((chunk_size - 1) / 2)
    right = x + int(chunk_size / 2)
    if top < 0:
        bottom = bottom - top
        top = 0
    if bottom > (matrix.shape[0] - 1):
        top = top - (bottom - matrix.shape[0] + 1)
        bottom = matrix.shape[0] - 1
    if left < 0:
        right = right - left
        left = 0
    if right > (matrix.shape[1] - 1):
        left = left - (right - matrix.shape[1] + 1)
        right = matrix.shape[1] - 1
    matrix_new = np.zeros((chunk_size, chunk_size, 3), matrix.dtype)
    matrix_new[:, :, :] = matrix[top:bottom + 1, left:right + 1, :]
    return matrix_new


def cut_image2chunks(image, chunks_center, label):
    matrix = image2matrix(image)
    chunk_w_letter = []
    for i, c in enumerate(chunks_center):
        matrix_new = cut_matrix(matrix, c[0], c[1])
        chunk_w_letter.append((matrix_new, label[i]))
    return chunk_w_letter




# ------------------------------------------
def mark_in_image(image, cls_list):
    matrix = image2matrix(image)
    matrix_new = np.zeros(matrix.shape, matrix.dtype)
    matrix_new[:, :, :] = matrix[:, :, :]
    image_b = to_binary_image(image)
    matrix_b = image2matrix(image_b)
    for c in cls_list:
        mid_h = int((min(c) + max(c)) / 2)
        head, rear = count_head_rear(matrix_b[:, mid_h])
        mid_v = int((head + rear) / 2)
        matrix_new[mid_v - 2: mid_v + 2, mid_h - 2: mid_h + 2, 0] = 255
        matrix_new[mid_v - 2: mid_v + 2, mid_h - 2: mid_h + 2, 1] = 0
        matrix_new[mid_v - 2: mid_v + 2, mid_h - 2: mid_h + 2, 2] = 0
    image_new = matrix2image(matrix_new)
    image_new.show()

def images_mark(image, label):
    cls = len(label)
    weight_in_col = flattening_with_weight(image)
    cls_list = clustering(weight_in_col, cluster=cls)
    mark_in_image(image, cls_list)
