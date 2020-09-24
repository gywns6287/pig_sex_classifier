from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import zscore

def CNN_LHJ(class_num, base_model = 'ResNet', input_shape = (224,224,3)):
    if base_model == 'ResNet':
        base = ResNet50(include_top = False, input_shape = input_shape)
    elif base_model == 'EfficientNet':
        base = EfficientNetB6(include_top= False, input_shape = input_shape)
    elif base_model == 'MobileNet':
        base = MobileNetV2(include_top= False, input_shape = input_shape)
    elif base_model == 'Xception':
        base = Xception(include_top= False, input_shape = input_shape)
    else:
        print('Error: Unknown base_model')
        return
    #
    inputs = Input(input_shape)
    feature = base(inputs)
    last_cnn = Conv2D(class_num,(3,3),activation = 'relu')(feature)
    #
    GAP = GlobalAveragePooling2D()(last_cnn)
    out =  Dense(class_num, activation = 'softmax')(GAP)
    return Model(inputs,out)

def make_CAM(model,img, out,class_indices,input_shape = (224,224,3)):
    weight_softmax = model.layers[-1].weights[0].numpy()
    CAM = Model(inputs = model.inputs, outputs = model.layers[-3].output)
    #
    img = np.array(img.resize(input_shape[:-1]))*1./255
    #
    feature = CAM.predict(img.reshape((1,)+input_shape))
    pred = model.predict(img.reshape((1,)+input_shape))
    idx = np.argmax(pred)
    prob = round(pred[0][idx],4)
    class_pre = class_indices[idx]
    #
    feature_map = np.zeros((feature.shape[1],feature.shape[2]))
    for w in range(len(weight_softmax[idx])):
        feature_map += feature[0][:,:,w] * weight_softmax[:,idx][w]
    feature_map = feature_map - np.min(feature_map)
    feature_map = feature_map/np.max(feature_map)
    feature_img = Image.fromarray(np.uint8(255* feature_map)).resize(input_shape[:-1])
    #
    plt.imshow(Image.fromarray((img*255).astype(np.uint8)))
    plt.imshow(feature_img, cmap='jet',alpha = 0.6)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out+'_'+class_pre+'_'+str(prob)+'.png')
'''
weight_softmax = model.layers[-1].weights[0].numpy()
CAM = Model(inputs = model.inputs, outputs = model.layers[-3].output)

c  = '수'
img_name = 'B-2020.03.20-09-05-08-0000257 C1.tif'
img_dir = test_path+'\\'+c+'\\'+img_name
img =image_crop(img_dir,crop_point,target_size = (224,224))

img = np.array(img.resize(input_shape[:-1]))*1./255
feature = CAM.predict(img.reshape((1,)+input_shape))
pred = model.predict(img.reshape((1,)+input_shape))
idx = np.argmax(pred)

for w in range(len(weight_softmax[idx])):
    #feature_map = feature[0][:,:,w]
    feature_map = feature[0][:,:,w]* weight_softmax[:,idx][w]
    print(feature_map)
    feature_map = feature_map - np.min(feature_map)
    feature_map = feature_map/np.max(feature_map)
    feature_img = Image.fromarray(np.uint8(255* feature_map)).resize(input_shape[:-1])
    #
    plt.imshow(Image.fromarray((img*255).astype(np.uint8)))
    plt.imshow(feature_img, cmap='jet',alpha = 0.6)
    plt.axis('off')
    plt.show()
'''
'''
from scipy.special import softmax
idx = 0
f_m = 0
for w in range(len(w_s[idx])):
    g_p = np.mean(feature[0][:,:,w])
    f_m += g_p*weight_softmax[idx][w]
    print(f_m)   
#np.array([-20.490785464644432, 22.2343083396554,25.223165281116962])
softmax(np.array([-20.490785464644432, 22.2343083396554,25.223165281116962]))
'''