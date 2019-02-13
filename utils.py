import numpy as np
import itertools
from matplotlib import pyplot as plt
import os, re
from PIL import Image

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_key(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def resize_image(path, new_path):
    filenames = []
    for root, dirname, filenames in os.walk(path):
        filenames.sort(key=natural_key)
        rootpath = root
        print(len(filenames))
        for item in filenames:
            if os.path.isfile(path+item):
                im = Image.open(path+item)
                f, e = os.path.splitext(item)
                imResize = im.resize((128, 128), Image.ANTIALIAS)
                imResize.save(new_path+f+'.jpg', 'JPEG', quality=90)

def get_filenames(path):
    filenames = []
    for root, dirname, filenames in os.walk(path):
        filenames.sort(key=natural_key)
        print(len(filenames))
        return filenames

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print('cm:', cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j]> thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')