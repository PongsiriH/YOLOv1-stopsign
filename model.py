import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Flatten, Dense, Dropout, Reshape
from tensorflow.keras import Sequential

lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

class YoloActivation(tf.keras.layers.Layer):
    def call(self, inputs):
        classes = tf.nn.softmax(inputs[..., 10:])
        bbox1, bbox2 = inputs[..., :5], inputs[..., 5:10]
        p1, xy1, wh1 = tf.nn.sigmoid(bbox1[..., :1]), tf.nn.sigmoid(bbox1[..., 1:3]), bbox1[..., 3:5]
        p2, xy2, wh2 = tf.nn.sigmoid(bbox2[..., :1]), tf.nn.sigmoid(bbox2[..., 1:3]), bbox2[..., 3:5]
        return tf.concat([p1, xy1, wh1, p2, xy2, wh2, classes], axis=-1)
        
def _renet_base(S: int, B: int, C: int, image_size: int=224) -> Sequential:
    image_size = image_size if isinstance(image_size, int) else (image_size[0], image_size[1])
    model = Sequential()
    model.add(
        tf.keras.applications
        .ResNet50V2(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3) )
    )

    model.add(Conv2D(512, 1, 1))
    model.add(BatchNormalization())
    model.add(lrelu)    
        
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(lrelu)

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(lrelu)

    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(lrelu)

    model.add(Dense(S * S * (B * 5 + C), kernel_initializer='he_normal')) 
    model.add(Reshape((S, S, B * 5 + C))) 
    model.add(YoloActivation())
    
    model.build(input_shape=(None, image_size, image_size, 3))
    for layer in model.layers[0].layers[:-80]:
        layer.trainable = False
    return model

if __name__ == '__main__':
    import numpy as np
    IMAGE_SIZE = 224
    images = np.random.normal(0, 1, (4, IMAGE_SIZE, IMAGE_SIZE, 3))
    
    model = _renet_base(7, 2, 10, image_size=IMAGE_SIZE)
    predictions = model.predict(images)
    
    print(images.shape, predictions.shape)