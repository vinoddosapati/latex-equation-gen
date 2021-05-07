import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO
import base64
from PIL import Image

from processing import get_components, process_image
from calculator import latexEval

# df = pd.read_csv('C:\\Users\\dpati\\OneDrive\\Desktop\\datapart\\data\\savedata\\dict.csv', delimiter=',', header=None)


df = pd.read_csv('dict.csv', delimiter=',', header=None)
key = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))


def classify(components):
    test = np.asarray([
        components[i]['pic'] for i in sorted(components.keys())
    ]).astype(np.float32)
    test = tf.reshape(test, [-1, 45, 45, 1])

    # ---------------------binary to image-----------------
    # # Convert the pixels into an array using numpy
    # temp_test = np.array(test[0])
    # temp_test *= 255
    # # r = len(temp_test)
    # # c = len(temp_test[0])
    # # for i in range(r):
    # #     for j in range(c):
    # #         if(temp_test[i][j] > 0):
    # #             print(temp_test[i][j])
    # imgg = np.reshape(temp_test, [45, 45])
    # array = np.array(imgg, dtype=np.uint8)
    # Use PIL to create an image from the new array of pixels
    # new_image = Image.fromarray(array)
    # new_image.save('new4.png')
    # ---------------------------------------------------

    print("test shape ", test.shape)
    new_model = tf.keras.models.load_model('seq_model_new.model')
    predict_results = new_model.predict(test)
    print("predit shape ", predict_results.shape)
    predicts = [np.argmax(predict_results[i]) for i in range(predict_results.shape[0])]
    print("predicts dict values =", predicts)

    # fig, axes = plt.subplots(2, int((len(components) + 1) / 2), figsize=(15, 5))
    # for i, (d, ax) in enumerate(zip(predicts, axes.ravel())):
    for i, d in enumerate(predicts):
        components[i + 1]['label'] = d
        components[i + 1]['output'] = key[d]
        # ax.imshow(components[i + 1]['pic'].reshape(45, 45), cmap='gray')
        # ax.axis('off')
        # ax.set_title('Classified as ' + key[d])
    # plt.show()
    return components


def assign_group(components, offset_threshold=3):
    heights = [[components[i]['tl'][0], components[i]['br'][0]] for i in components]
    print("heigths tb : ", heights)
    groups = [heights[0]]
    for height in heights[1:]:
        print(groups)
        if height[0] + offset_threshold < groups[-1][1]:
            groups[-1][1] = max(height[1], groups[-1][1])
        else:
            # groups[-1][1] = max(height[1], groups[-1][1])
            groups.append(height)
    for i in components:
        for group in groups:
            if group[0] < components[i]['tl'][0] + offset_threshold < group[1]:
                components[i]['group'] = group
    return components, groups


def detect_script(components, groups):
    for g in groups:
        bottoms = [components[i]['br'][0] for i in sorted(components.keys()) if components[i]['group'] == g]
        tops = [components[i]['tl'][0] for i in sorted(components.keys()) if components[i]['group'] == g]
        bottoms_mean = np.mean(bottoms)
        bottoms_std = np.std(bottoms)
        tops_mean = np.mean(tops)
        tops_std = np.std(tops)

        if len(bottoms) == 1: continue
        for i in components:
            if components[i]['group'] == g:
                z_score_bottom = (bottoms_mean - components[i]['br'][0]) / bottoms_std
                z_score_top = (components[i]['tl'][0] - tops_mean) / tops_std
                s = z_score_bottom - z_score_top
                print(s, "--", i, " component : ", components[i]['output'], "zbtm ",z_score_bottom, " ztop ", z_score_top, " zscore :", s)
                if s > 2.35:
                    components[i]['sup'] = True
                # elif s < -2.5:
                    # components[i]['sub'] = True
    return components

def finffrac(components, groups):
    l_order = sorted(components.keys(), key=lambda x: components[x]['tl'][1])
    for i in range(len(l_order)):
        if (components[l_order[i]]['output'] == '-'):
            idx = i
            minus_range = [components[l_order[idx]]["tl"][1], components[l_order[idx]]["br"][1]]
            comp_range = [components[l_order[i+1]]["tl"][1], components[l_order[i+1]]["br"][1]]
            if(minus_range[0] <= comp_range[0] <= comp_range[1] <= minus_range[1] and components[l_order[idx]]['group'] == components[l_order[i+1]]['group']):
                print("old ", components[l_order[idx]]['output'])
                components[l_order[idx]]['output'] = '\\frac'
                print("new ", components[l_order[idx]]['output'])
    return components


def neworder(components, groups):
    components = finffrac(components, groups)
    print("new com ", components)
    l_order = sorted(components.keys(), key=lambda x: components[x]['tl'][1])
    order = [components[i]["output"] for i in l_order]
    print("lorder ",l_order)
    print("order ", order)
    num = []
    denom = []
    frac = 0
    s = []
    idx = 0
    for i in l_order:
        if(components[i]["output"] == '\\frac' and frac==0):
            s.append(i)
            idx = i
            frac = 1
            continue
        elif(frac and components[idx]['group'] == components[i]['group']):
            minus_range = [components[idx]["tl"][1], components[idx]["br"][1]]
            comp_range = [components[i]["tl"][1], components[i]["br"][1]]
            if(minus_range[0] <= comp_range[0] <= comp_range[1] <= minus_range[1] and components[idx]['group'] == components[i]['group']):
                components[i]['sup'] = False
                components[i]['sub'] = False
                print("its frac ", components[i]["output"])
                if(components[idx]["tl"][0] < components[i]["tl"][0]):
                    print("its denmo ", components[i]["output"])
                    denom.append(i)
                else:
                    print("its num ", components[i]["output"])
                    num.append(i)
            else:
                new_num = [components[j]["output"] for j in num]
                new_den = [components[j]["output"] for j in denom]
                print("new num ", new_num, " new den ", new_den)
                frac = 0
                components[num[0]]['output'] = '{' + components[num[0]]['output']
                components[num[-1]]['output'] = components[num[-1]]['output'] + '}'
                components[denom[0]]['output'] = '{' + components[denom[0]]['output']
                components[denom[-1]]['output'] = components[denom[-1]]['output'] + '}'

                s += num
                s += denom
                num = []
                denom = []
                print("terminate,", components[i]["output"])
                s.append(i)
        else:
            s.append(i)
    if(len(num) or len(denom)):
        frac = 0
        components[num[0]]['output'] = '{' + components[num[0]]['output']
        components[num[-1]]['output'] = components[num[-1]]['output'] + '}'
        components[denom[0]]['output'] = '{' + components[denom[0]]['output']
        components[denom[-1]]['output'] = components[denom[-1]]['output'] + '}'
        new_num = [components[j]["output"] for j in num]
        new_den = [components[j]["output"] for j in denom]
        print("new num ", new_num, " new den ", new_den)
        s += num
        s += denom
        num = []
        denom = []
    print("s =", s)
    print("num =",num,"denom =" ,denom)
    return s





def construct_latex(components, groups):
    lr_order = sorted(components.keys(), key=lambda x: components[x]['tl'][1])
    order = [components[i]["output"] for i in lr_order]
    print("order ",order)
    lr_order = neworder(components, groups)
    order = [components[i]["output"] for i in lr_order]
    print("after order ", order)
    vsep = {tuple(group): [] for group in groups}
    print(type(vsep))
    print(vsep)
    MODE_SUP = set()
    MODE_SUB = set()
    MODE_SQRT = {}


    for l in lr_order:
        t, left = components[l]['tl']
        b, right = components[l]['br']
        for g in vsep:
            print(g)
            if g[0] <= t <= b <= g[1]:

                if g in MODE_SQRT and left > MODE_SQRT[g]:
                    vsep[g].append('}')
                    del MODE_SQRT[g]
                if g in MODE_SUP and not components[l]['sup']:
                    vsep[g].append('}')
                    MODE_SUP.remove(g)
                if g in MODE_SUB and not components[l]['sub']:
                    vsep[g].append('}')
                    MODE_SUB.remove(g)
                if g not in MODE_SUP and components[l]['sup']:
                    vsep[g].append('^{')
                    MODE_SUP.add(g)
                if g not in MODE_SUB and components[l]['sub']:
                    vsep[g].append('_{')
                    MODE_SUB.add(g)
                vsep[g].append(components[l]['output'] + ' ')
                if components[l]['output'] == '\\sqrt':
                    MODE_SQRT[g] = right
                    vsep[g].append('{')
                break

    print("MODE SQRT, ", print(MODE_SQRT))
    for i in MODE_SQRT:
        vsep[i].append('}')
    for j in MODE_SUP:
        vsep[j].append('}')
    for g in vsep:
        vsep[g] = ''.join(vsep[g])

    print("after script",vsep)

    if len(vsep) == 3:
        first_g, _, last_g = list(sorted([g for g in vsep], key=lambda g: g[0]))
        final = '\\frac{' + vsep[first_g] + '}{' + vsep[last_g] + '}'
    else:
        final = list(vsep.values())[0]
    final = final.replace(" ", "")
    final = final.replace("\lambda", "\lambda ")
    return final

#

# # read in image
# # file_name_X = '../imageprocess/download (2).png'
# # X = (mpimg.imread(file_name_X)[:, :, 0:3]).reshape(-1, 3)
# # new_shape = list(mpimg.imread(file_name_X).shape)
# # new_shape[2] = 3
# # X = X.reshape(new_shape)
# # print(type(X))
# data='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5uUgYIE/AvhAjraVbVKslMD2KE7B2oM6+3QSsMDLAD6z0eyLQlVRPlKwKgpWz6Za4HmkP75CsaosqShYlQWsd3PH/a+vBcy+KE4NJA8Fq4Eg0gUS6IUABauXSNNPEmiAAAWrgSDSBRLohQAFq5dI008SaIAABauBINIFEuiFAAWrl0jTTxJogAAFq4Eg0gUS6IUABauXSNNPEmiAAAWrgSDSBRLohQAFq5dI008SaIAABauBINIFEuiFAAWrl0jTTxJogAAFq4Eg0gUS6IUABauXSNNPEmiAAAWrgSDSBRLohQAFq5dI008SaIAABauBINIFEuiFAAWrl0jTTxJogAAFq4Eg0gUS6IUABauXSNNPEmiAAAWrgSDSBRLohQAFq5dI008SaIAABauBINIFEuiFAAWrl0jTTxJogAAFq4Eg0gUS6IUABauXSNNPEmiAAAWrgSDSBRLohQAFq5dI008SaIAABauBINIFEuiFAAWrl0jTTxJogAAFq4Eg0gUS6IUABauXSNNPEmiAAAWrgSDSBRLohQAFq5dI008SaIAABauBINIFEuiFAAWrl0jTTxJogAAFq4Eg0gUS6IUABauXSNNPEmiAAAWrgSDSBRLohQAFq5dI008SaIAABauBINIFEuiFAAWrl0jTTxJogAAFq4Eg0gUS6IUABauXSNNPEmiAAAWrgSDSBRLohQAFq5dI008SaIDA/wC6duDE9qt18wAAAABJRU5ErkJggg=='
# file_name_X = Image.open(BytesIO(base64.b64decode(data)))
# X_processed = cv2.cvtColor(np.array(file_name_X), cv2.COLOR_BGR2GRAY)
#
#
# # X_processed = process_image(file_name_X)
#
# # plt.imshow(X)
# # plt.axis('off')
# # plt.show()
# #
# plt.imshow(X_processed, cmap='gray')
# plt.axis('off')
# plt.show()
# print("processed image shape ", X_processed.shape)
# _, labels = cv2.connectedComponents(X_processed)
#
# plt.imshow(labels)
# plt.axis('off')
# plt.show()
# print("labeled image shape ", labels.shape)
# components = get_components(labels)
# print("Components1: ", components)
# components = classify(components)
# print("Components2: ", components)
# components, groups = assign_group(components)
# print("Components3: ", components)
# print("Groups : ", groups)
# components = detect_script(components, groups)
# print("Components4: ", components)
# expression = construct_latex(components, groups)
# print("expression : ", expression)
# #
# # fig, ax = plt.subplots()
# # ax.text(0.5, 0.5, expression, size=50)
# # ax.axis('off')
# # plt.show()
#


def expLatex(data):
    # data='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'
    file_name_X1 = Image.open(BytesIO(base64.b64decode(data)))
    X_processed1 = cv2.cvtColor(np.array(file_name_X1), cv2.COLOR_BGR2GRAY)

    # X_processed = process_image(file_name_X)

    # plt.imshow(X)
    # plt.axis('off')
    # plt.show()
    #
    # plt.imshow(X_processed1, cmap='gray')
    # plt.axis('off')
    # plt.show()
    print("processed image shape ", X_processed1.shape)
    _, labels1 = cv2.connectedComponents(X_processed1)

    # plt.imshow(labels1)
    # plt.axis('off')
    # plt.show()
    print("labeled image shape ", labels1.shape)
    components1 = get_components(labels1)
    print("Components1: ", components1)
    components2 = classify(components1)
    print("Components2: ", components2)
    components3, groups1 = assign_group(components2)
    print("Components3: ", components3)
    print("Groups : ", groups1)
    components4 = detect_script(components3, groups1)
    print("Components4: ", components4)
    expression1 = construct_latex(components4, groups1)
    # expression1 = ""
    print("expression : ", expression1)
    # fig, ax = plt.subplots()
    # ax.text(0.5, 0.5, expression1, size=50)
    # ax.axis('off')
    # plt.show()
    # expression1 = expression1 + "OUTPUT" + str(latexEval(expression1))
    return expression1