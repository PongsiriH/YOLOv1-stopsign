import tensorflow as tf
from model import (
    YoloActivation
)
from utils import (
    yolo2xywh,
    YOLOv1DataLoader
)
from plot_utils import (
    plt,
    plot_labels,
    get_best_prediction,
    apply_confidence_thresh,
    components2yolo,
    LABELS_LIST_STOPSIGN, LABELS_DICT_STOPSIGN
)

import matplotlib.pyplot as plt

PATH_data = 'D:\\Year 3\\PyTorch tutorials\\StopSign\\data'
IMAGE_DIR = PATH_data + '\\images'
LABEL_DIR = PATH_data + '\\labels'
IMAGE_SIZE = 224
NUM_GRIDS = 7
NUM_BBOXES = 2
NUM_CLASSES = 4
LAMBDA_COORD = 5
LAMBDA_OBJ = 1
LAMBDA_NOOBJ = 0.5
LAMBDA_CLS = 1
EPOCHS = 100
LEARNING_RATE = 1e-2
BATCH_SIZE = 1
image_size = (IMAGE_SIZE, IMAGE_SIZE)

val_loader = YOLOv1DataLoader(IMAGE_DIR, LABEL_DIR, image_size, 
                            num_grids=NUM_GRIDS, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE,
                            image_format='.png')
images, labels = val_loader.__getitem__(0)

model = tf.keras.models.load_model('save_stop_sign/47epochs.h5', custom_objects={'YoloActivation': YoloActivation})
predictions = model.predict(images)

# print('shapes: ', images.shape, labels.shape, predictions.shape)
idx = 0
image = images[idx]
label = labels[idx]
prediction = predictions[idx]


print('actual')
plot_labels(image, yolo2xywh(label), class_labels=LABELS_DICT_STOPSIGN, num_grids=NUM_GRIDS)
best_conf, best_bbox, cls = get_best_prediction(prediction)

# print('best_conf: before thresh', best_conf.squeeze(-1))

best_conf = apply_confidence_thresh(best_conf)
# print('best_conf: after thresh', best_conf.squeeze(-1))

yolo_prediction = components2yolo(best_bbox, best_conf, cls)
print(yolo_prediction.shape)

print('predicted')
plot_labels(image, yolo2xywh(yolo_prediction), class_labels=LABELS_DICT_STOPSIGN, num_grids=NUM_GRIDS)
print('closing...')