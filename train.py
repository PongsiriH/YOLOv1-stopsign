# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from loss import YOLOv1Loss
from model import _renet_base, YoloActivation
from utils import (
    YOLOv1DataLoader,
    yolo2xywh
)

import os, pickle

PATH_data = 'D:\\Year 3\\PyTorch tutorials\\StopSign\\data'
IMAGE_DIR = PATH_data + '\\images'
LABEL_DIR = PATH_data + '\\labels'
IMAGE_SIZE = 224
NUM_GRIDS = 7
NUM_BBOXES = 2
NUM_CLASSES = 4

LAMBDA_COORD = 2
LAMBDA_OBJ = 1
LAMBDA_NOOBJ = 0.5
LAMBDA_CLS = 1

EPOCHS = 1000
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 0.005
BATCH_SIZE = 16
image_size = (IMAGE_SIZE, IMAGE_SIZE)

def train(save_to):
    save_to = f'saved_models\\{save_to}\\'
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    val_loader = YOLOv1DataLoader(IMAGE_DIR, LABEL_DIR, image_size, 
                            num_grids=NUM_GRIDS, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE,
                            image_format='.png')
    model = _renet_base(NUM_GRIDS, 2, NUM_CLASSES, image_size=IMAGE_SIZE)
    loss_fn = YOLOv1Loss(NUM_GRIDS, NUM_CLASSES, NUM_BBOXES, LAMBDA_COORD, LAMBDA_OBJ, LAMBDA_NOOBJ, LAMBDA_CLS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    @tf.function(jit_compile=True)
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            labels = tf.cast(labels, predictions.dtype)
            loss, loss_components = loss_fn(labels, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, loss_components

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(val_loader):
            loss, loss_components = train_step(images, labels)
            total_loss += loss.numpy()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(val_loader)}], Loss: {loss.numpy():.8f}, lr: {optimizer.learning_rate.numpy()}, {[f'{key}: {val:.8f}' for key, val in loss_components.items()]}")

        optimizer.learning_rate.assign(optimizer.learning_rate * (1-WEIGHT_DECAY))
        avg_loss = total_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")
        
        if epoch % 20 == 0:
            print(f'saving model as {save_to}{EPOCHS}epochs.h5')
            model.save(f'{save_to}{epoch+1}epochs.h5')

    print(f'saving model as {save_to}last_epochs.h5')
    model.save(f'{save_to}last_epochs.h5')

def resume_train(model_path, save_to, intial_epoch):
    model_path = f'saved_models\\{model_path}'
    save_to = f'saved_models\\{save_to}\\'
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    val_loader = YOLOv1DataLoader(IMAGE_DIR, LABEL_DIR, image_size, 
                            num_grids=NUM_GRIDS, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE,
                            image_format='.png')
    model = tf.keras.models.load_model(model_path, custom_objects={'YoloActivation': YoloActivation})
    loss_fn = YOLOv1Loss(NUM_GRIDS, NUM_CLASSES, NUM_BBOXES, LAMBDA_COORD, LAMBDA_OBJ, LAMBDA_NOOBJ, LAMBDA_CLS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    @tf.function(jit_compile=True)
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss, loss_components = loss_fn(labels, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, loss_components

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(val_loader):
            loss, loss_components = train_step(images, labels)
            total_loss += loss.numpy()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(val_loader)}], Loss: {loss.numpy():.8f}, lr: {optimizer.learning_rate.numpy()}, {[f'{key}: {val:.8f}' for key, val in loss_components.items()]}")

        optimizer.learning_rate.assign(optimizer.learning_rate * (1-WEIGHT_DECAY))
        avg_loss = total_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.4f}")
        
        if epoch % 20 == 0:
            print(f'saving model as {save_to}{epoch+1}epochs.h5')
            model.save(f'{save_to}{epoch+1}epochs.h5')

    print(f'saving model as {save_to}last_epochs.h5')
    model.save(f'{save_to}last_epochs.h5')

def test():
    import numpy as np
    import matplotlib.pyplot as plt
    from model import YoloActivation
    from plot_utils import (
        LABELS_DICT_STOPSIGN,
        plot_labels,
        get_best_prediction,
        apply_confidence_thresh,
        components2yolo
    )
    
    PATH_MODEL = f'saved_models\\{SAVE_TO}\\last_epochs.h5'
    with open(f'saved_models\\SAVE_TO\\data', 'rb') as f:
        images, labels = pickle.load(f)
    print(images.shape, labels.shape)
    
    model = tf.keras.models.load_model(PATH_MODEL, custom_objects={'YoloActivation': YoloActivation})
    predictions = model.predict(images)
    
    idx = 0
    for idx in range(len(labels)):
        image = np.array(images[idx])
        label = np.array(labels[idx])
        prediction = np.array(predictions[idx])
        
        best_conf, best_bbox, cls = get_best_prediction(prediction)
        print('best_conf:\n', best_conf.squeeze(-1))
        best_conf = apply_confidence_thresh(best_conf, conf_thresh=0.5)
        print('best_conf:\n', best_conf.squeeze(-1))
        print('label_conf:\n', label[..., 0].astype('int'))
        
        yolo_prediction = components2yolo(best_bbox, best_conf, cls)
        
        print('actual :', len(yolo2xywh(label)), yolo2xywh(label))
        print('predict:', len(yolo2xywh(yolo_prediction)), yolo2xywh(yolo_prediction))
        
        plt.subplot(1, 2, 1)
        plot_labels(image, yolo2xywh(label), class_labels=LABELS_DICT_STOPSIGN, num_grids=NUM_GRIDS,
                        title='actual')
        
        plt.subplot(1, 2, 2)
        plot_labels(image, yolo2xywh(yolo_prediction), class_labels=LABELS_DICT_STOPSIGN, num_grids=NUM_GRIDS,
                        title='predicted')
        plt.savefig(f'saved_models\\train_full_after_warmup1\\{idx}.png')
        plt.show()
if __name__ == '__main__':
    # SAVE_TO = 'warm_up2'
    # train(save_to=SAVE_TO)
    
    SAVE_TO = 'train_full_after_warmup1'
    # resume_train('warm_up\\last_epochs.h5',
    #              SAVE_TO,
    #              intial_epoch=0)

    # SAVE_TO = 'train_full_after_warmup2'
    # resume_train('warm_up2\\last_epochs.h5',
    #              SAVE_TO,
    #              intial_epoch=0)
     
    test()
