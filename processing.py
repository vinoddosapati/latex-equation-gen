import numpy as np
import cv2


def process_image(file_name_X, dilation_kernel=None, dilation_iterations=3):
    # binary
    X_gray_smooth = cv2.cvtColor(cv2.imread(file_name_X).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # X_gray_smooth = cv2.cvtColor(np.array(file_name_X), cv2.COLOR_BGR2GRAY)

    # remove initial noise and smoothen light
    X_gray_smooth = cv2.GaussianBlur(X_gray_smooth, (11, 11), 0)

    # threshold
    X_gray_smooth = cv2.adaptiveThreshold(X_gray_smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # blur
    X_gray_smooth = cv2.blur(X_gray_smooth, (3, 3))

    # thrershold
    X_gray_smooth = cv2.threshold(X_gray_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Median blur
    # X_gray_smooth = cv2.medianBlur(X_gray_smooth, 17)

    # threshold
    X_bw = 1 - np.round(X_gray_smooth / 255)

    # Dilate
    if not dilation_kernel: dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    X_bw = cv2.dilate(X_bw, dilation_kernel, iterations=dilation_iterations)

    return X_bw.astype(np.uint8)


def pad(arr, pad_width):
    arr_new = np.hstack([np.zeros([arr.shape[0], pad_width]), arr])
    arr_new = np.hstack([arr_new, np.zeros([arr_new.shape[0], pad_width])])
    arr_new = np.vstack([np.zeros([pad_width, arr_new.shape[1]]), arr_new])
    arr_new = np.vstack([arr_new, np.zeros([pad_width, arr_new.shape[1]])])
    return arr_new


def square(arr, pad_width, top, left, bottom, right, i):
    print("component ", i, " shape ", bottom-top, "x", right-left)
    print(arr.shape, "--", np.unique(arr))
    arr[arr != i] = 0
    arr = arr//i
    print(arr.shape, "--", np.unique(arr))
    arr_square = arr[top :bottom + pad_width + pad_width, left :right + pad_width + pad_width]
    print(arr_square.shape[0], "----", arr_square.shape[1])
    diff = abs(arr_square.shape[1] - arr_square.shape[0])
    pad = diff // 2
    if arr_square.shape[0] < arr_square.shape[1]:
        arr_square = np.vstack([np.zeros([pad, arr_square.shape[1]]), arr_square])
        arr_square = np.vstack([arr_square, np.zeros([pad + (diff % 2 == 1), arr_square.shape[1]])])
    elif arr_square.shape[0] >= arr_square.shape[1]:
        arr_square = np.hstack([np.zeros([arr_square.shape[0], pad]), arr_square])
        arr_square = np.hstack([arr_square, np.zeros([arr_square.shape[0], pad + (diff % 2 == 1)])])
    return arr_square


def erode(arr, erosion_percent):
    for dim in range(1, 12):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dim, dim))
        erosion = cv2.erode(arr, kernel, iterations=1)
        if np.sum(erosion) / np.sum(arr) < erosion_percent:
            break
    dim -= 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (dim, dim))
    erosion = cv2.erode(arr, kernel, iterations=1)
    return erosion


def get_components(labels, pad_width=3, erosion_percent=0.4, show=True):
    print("components found in image ", len(np.unique(labels)) - 1)
    components = {i: {'label': None,
                      'output': None,
                      'tl': None,
                      'br': None,
                      'pic': None,
                      'group': None,
                      'sup': False,
                      'sub': False,
                      'num': False,
                      'deno': False}
                  for i in range(1, len(np.unique(labels)))}
    # fig, axes = plt.subplots(2, int((len(components) + 1) / 2), figsize=(15, 5))
    # for i, ax in zip(sorted(components.keys()), axes.ravel()):
    for i in sorted(components.keys()):
        label = labels.copy()
        label_padded = pad(label, pad_width)
        print("After padding with 3 to image shape ", label_padded.shape)

        # get dimensions of component
        # label |= 0
        # print("label  =", labels, "type =", type(labels))
        # print("unique val : ", np.unique(labels))
        xs, ys = np.where(label == i)
        top, bottom, left, right = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
        print("For component ", i)
        print("Top: ", top, " bottom: ", bottom, " left: ", left, " right: ", right)
        components[i]['tl'] = (top, left)
        components[i]['br'] = (bottom, right)

        # square and resize
        # label_square = labels.copy()
        # print("label  =", label_padded, "type =", type(label_padded))
        # print("unique val : ", np.unique(label_padded), "typr", type(np.unique(label_padded)[0]))
        print(",,,", label_padded.shape)
        label_square = square(label_padded, pad_width, top, left, bottom, right, i)
        print("---", label_square.shape)
        label_square = cv2.resize(label_square, (45, 45))
        label_square[label_square != 0] = 1
        print("after resize ", label_square.shape)
        # print("i =", i, "label square =", label_square, "type =", type(label_square))
        # print("unique val : ", np.unique(label_square))
        # label_squares = label_squares | 0
        # label_square[label_squares] = 1

        # erode based on size
        label_eroded = erode(label_square, erosion_percent)
        components[i]['pic'] = label_eroded.ravel()
        # plt.imshow(label_eroded, cmap='gray')
        # plt.axis('off')
        # plt.show()
    return components


# # read in image
# file_name_X = 'test4.jpg'
# X = (mpimg.imread(file_name_X)[:, :, 0:3]).reshape(-1, 3)
# new_shape = list(mpimg.imread(file_name_X).shape)
# new_shape[2] = 3
# X = X.reshape(new_shape)
#
# X_processed = process_image(X)
#
# # plt.imshow(X)
# # plt.axis('off')
# # plt.show()
# #
# # plt.imshow(X_processed, cmap='gray')
# # plt.axis('off')
# # plt.show()
# print("processed image shape ", X_processed.shape)
# _, labels = cv2.connectedComponents(X_processed)
#
# plt.imshow(labels)
# plt.axis('off')
# plt.show()
# print("labeled image shape ", labels.shape)
# for l in range(labels.shape[1]):
#     print(labels[90][l], end=" ")
# print(" ")
# components = get_components(labels)
# print("Components: ", components)
