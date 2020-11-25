
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train")
parser.add_argument("--test")
parser.add_argument("--out",default='.')
parser.add_argument("--device", default = '1')
args = parser.parse_args()

device = args.device
train_path = args.train
test_path = args.test
out = args.out
import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=device

from CAM_LHJ import *
from data_loader import *
from tensorflow.keras.optimizers import Adam
import time
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from PIL import Image
#python solution.py --train ..\\00.data\\train --test ..\\00.data\\test --out results --device 0
#################################
'''
device = '0'
train_path = '..\\00.data\\train'
test_path = '..\\00.data\\test'
out = 'results'
'''
################################


epochs = 100
batch_size = 16
input_shape = (224,224,3)
crop_point = (0.4,0.4)
lr = 1e-04
decay = 0.1
r = 30
class_num = 3
#################################

#model load
model = CNN_LHJ(class_num, base_model = 'ResNet', input_shape = input_shape)
model.compile(Adam(learning_rate = lr),"categorical_crossentropy",metrics = ['accuracy'])


#data load
train = crop_loader(train_path, crop_point,target_size = input_shape[:-1], batch_size = batch_size , mode = 'train')
test = crop_loader(test_path, crop_point,target_size = input_shape[:-1], batch_size = batch_size , mode = 'test')

#training
lr_step = step_decay(lr,decay,round = r)
lr_scheduler = LearningRateScheduler(lr_step)
csv_logger = CSVLogger(out+'\\'+"train_history.txt", append=True, separator='\t')
model_checkpoint = ModelCheckpoint(out+'\\'+"weight.h5", 
                                                      monitor='loss',
                                                      verbose=1, 
                                                      save_best_only=True)
s_time = time.time()
model.fit(train,
            steps_per_epoch = len(train.geninfo),
            epochs = epochs, 
            callbacks= [model_checkpoint,csv_logger,lr_scheduler],
            validation_data = test,
            validation_steps = len(test.geninfo))
e_time = time.time()
#model.load_weights(out+'\\weight.h5')
##############################################



##test
test = crop_loader(test_path, crop_point,target_size = input_shape[:-1], batch_size = batch_size , mode = 'test')
predictor = test_prediction(model, test, out)
summary_dic = {'Learning_time': e_time - s_time,
               'Accuracy': predictor.acc,
               'F1 score':predictor.f1,
               'Learning_rate':lr,
               'epochs': epochs,
               'batch_size': batch_size,
               'input_shape': input_shape}
file_save(summary_dic, test.geninfo.filenames, predictor.pre_class, predictor.true_class, predictor.prob, out)
#####
#CAM
import random
class_dic = {v:k for k,v in test.geninfo.class_indices.items()}
for _ in range(5):
    for c in test.geninfo.class_indices.keys():
        img_name = random.sample(os.listdir('{0}\\{1}'.format(test_path,c)), 1)[0]
        img_dir = test_path+'\\'+c+'\\'+img_name
        img =image_crop(img_dir,crop_point,target_size = (224,224))
        make_CAM(model, img, out+'\\'+img_name+'_'+c,class_dic,input_shape = input_shape)



