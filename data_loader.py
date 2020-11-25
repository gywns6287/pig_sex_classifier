import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
def data_loader(path, target_size = (224,224), batch_size = 16,  mode = 'train'):
    if mode == 'train':
        datagen = ImageDataGenerator(rescale = 1./255,
                                rotation_range=0.2,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                shear_range=0.05,
                                zoom_range=0.05,
                                horizontal_flip=True,
                                fill_mode='nearest')
        return datagen.flow_from_directory(path,
                                        target_size = target_size,
                                        batch_size = batch_size)
    else:
        datagen = ImageDataGenerator(rescale = 1./255)
        return datagen.flow_from_directory(path,
                                        target_size = target_size,
                                        batch_size = batch_size,
                                        shuffle = False)

class test_prediction():
    def __init__(self, model,gen,out):
        class_dict = {v:k for k,v in gen.geninfo.class_indices.items()}
        #
        pre = model.predict(gen,steps = len(gen.geninfo)) 
        self.pre_class = np.array([class_dict[i] for i in np.argmax(pre, axis = 1)])
        self.true_class = np.array([class_dict[i] for i in gen.geninfo.classes])
        self.prob = pre[(range(len(pre)),np.argmax(pre, axis = 1))] 
        #
        from sklearn.metrics import f1_score
        self.f1 = round(f1_score(self.pre_class,self.true_class,average = 'macro'),4)
        self.acc = round(sum(self.pre_class ==self. true_class)/len(self.true_class),4)


def file_save(summary_dict, filenames, pred, true, prob, out):
    with open(out+'\\'+'pred.sol','w') as sol:
        sol.write('FILE\tTRUE\tPRED\tprob\n')
        for f, t, p, pr in zip(filenames, true, pred, prob):
            sol.write('\t'.join([f, t, p, str(pr)])+'\n')
    #
    with open(out+'\\'+'summary.txt','w') as save:
        for k,v in summary_dict.items():
            save.write('{0}: {1}\n'.format(k,str(v)))

def image_crop(img_dir,crop_point,target_size = (224,224)):
    #
    w = int(target_size[0]/(1-crop_point[0]))
    h = int(target_size[1]/(1-crop_point[1]))
    img = np.array(Image.open(img_dir).resize((w,h)))
    #
    w -= target_size[0]
    h -= target_size[1]
    return Image.fromarray(img[h:,w:,:]) 

class crop_loader():
    #
    def __init__(self, path, crop_point,target_size = (224,224),batch_size = 16,  mode = 'train'):
        start_size = (int(target_size[0]/(1-crop_point[0])),
                    int(target_size[1]/(1-crop_point[1])))
        self.target_size = target_size
    #
        if mode == 'train':
            datagen = ImageDataGenerator(rescale = 1./255,
                                    rotation_range=0.2,
                                    width_shift_range=0.05,
                                    height_shift_range=0.05,
                                    shear_range=0.05,
                                    zoom_range=0.05,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
            self.geninfo = datagen.flow_from_directory(path,
                                    target_size = start_size,
                                     batch_size = batch_size)
        else:
            datagen = ImageDataGenerator(rescale = 1./255)
            self.geninfo = datagen.flow_from_directory(path,
                                        target_size = start_size,
                                        batch_size = batch_size,
                                        shuffle = False)
 #   
    def __iter__(self):
        return self
#
    def __next__(self):
        x,y = next(self.geninfo)
        h = x.shape[1] - self.target_size[1]
        w = x.shape[2] - self.target_size[0]
        x_ = x[:,h:,w:,:]
        return x_, y 

def step_decay(lr0,decay,round = 20):
    def step_decay_fn(epoch):
        R = (epoch)//round + 1
        new_lr = lr0 * decay**(R-1)
        print('lr: {0}'.format(new_lr))
        return new_lr
    return step_decay_fn  
