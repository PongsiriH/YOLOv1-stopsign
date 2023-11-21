import os, cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import albumentations as A
from albumentations.core.composition import Compose, OneOf

def read_txt_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                annotations.append((class_id, x_center, y_center, width, height))
    return annotations

def xyxy2xywh(bboxes):
    out = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        out.append([x_center, y_center, width, height])
    return out

def xywh2xyxy(bboxes):
    out = []
    for bbox in bboxes:
        x, y, w, h = bbox
        x1 = max(0, min(1, x - w / 2))
        y1 = max(0, min(1, y - h / 2))
        x2 = max(0, min(1, x + w / 2))
        y2 = max(0, min(1, y + h / 2))
        out.append([x1, y1, x2, y2])
    return out

def xywh2yolo(labels, num_grids, num_classes, relative_to_grids : bool=True):
    if isinstance(labels, tuple) and len(tuple) == 4:
        labels = [labels]

    grid = np.zeros((num_grids, num_grids, 5 + num_classes))

    for label in labels:
        class_id, x_center, y_center, width, height = label
        grid_x = int(x_center * num_grids)
        grid_y = int(y_center * num_grids)

        x_cell = x_center * num_grids - grid_x
        y_cell = y_center * num_grids - grid_y
        if relative_to_grids:
            width *= num_grids
            height *= num_grids

        if 0 <= grid_x < num_grids and 0 <= grid_y < num_grids:
            if grid[grid_y, grid_x, 0] == 0:
                grid[grid_y, grid_x, 1:5] = [x_cell, y_cell, width, height]
                grid[grid_y, grid_x, 0] = 1  # Objectness score
                grid[grid_y, grid_x, 5 + class_id] = 1  # One-hot encoded class label
            
    return grid

def yolo2xywh(yolo_labels, threshold=0.0):
    """
    Convert YOLO label to a list of bounding boxes in normalized xywh format.

    :param yolo_labels: A numpy array of shape (S, S, C+5).
    :param threshold: Threshold for objectness score to consider a detection.
    :return: List of tuples (x_center, y_center, width, height).
    """
    S = yolo_labels.shape[0]  # Grid size
    xywh_list = []

    for grid_y in range(S):
        for grid_x in range(S):
            cell_info = yolo_labels[grid_y, grid_x]
            objectness_score = cell_info[0]
            
            if objectness_score > threshold:
                # Extract bounding box information
                x_cell, y_cell, width, height = cell_info[1:5]
                class_id = cell_info[5:].argmax()
                

                # Convert from grid-relative to image-relative coordinates
                x_center = (grid_x + x_cell) / S
                y_center = (grid_y + y_cell) / S

                xywh_list.append((class_id, x_center, y_center, width, height))

    return xywh_list

class YOLOv1DataLoader(Sequence):
    def __init__(self, image_dir, label_dir, image_size,
                 num_grids, num_classes, batch_size, normalize=True, augment='default',
                 image_format='.jpg'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.S = num_grids
        self.C = num_classes
        self.image_format = image_format
        self.normalize = normalize
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(self.image_format)]
        self.label_paths = [p.replace('images', 'labels').replace(self.image_format, '.txt') for p in self.image_paths]
        
        if isinstance(augment, A.Compose):
            self.augment = augment
        elif augment == 'default':
            self.augment = A.Compose([
                A.HorizontalFlip(p=0.05),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=(3, 7), p=0.5),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), shadow_dimension=5, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.4))
        else:
            self.augment = None
            
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_label_paths = self.label_paths[index * self.batch_size:(index + 1) * self.batch_size]
        
        images = []
        yolo_labels = np.zeros((self.batch_size, self.S, self.S, 5 + self.C))
        
        for i, (img_path, label_path) in enumerate(zip(batch_image_paths, batch_label_paths)):
            # Load image and label
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            labels = read_txt_annotations(label_path)
            
            bboxes = [np.clip(label[1:], 0, 1) for label in labels]
            bboxes = xyxy2xywh(xywh2xyxy(bboxes)) # to clip the bounding boxes
            class_labels = [label[0] for label in labels]
            
            # Apply augmentations
            augmented = self.augment(image=image, bboxes=bboxes, class_labels=class_labels)
            image = augmented['image']
            augmented_bboxes = augmented['bboxes']
            
            # Resize image to target size
            image = cv2.resize(image, self.image_size)
            updated_labels = [(class_labels[j], *bbox) for j, bbox in enumerate(augmented_bboxes)]

            # Convert labels to YOLO format
            yolo_label = xywh2yolo(updated_labels, self.S, self.C, relative_to_grids=True)
            
            images.append(image)
            yolo_labels[i] = np.array(yolo_label, dtype='float32')

        images = np.array(images, dtype='float32')
        
        if self.normalize:
            images = images / 255.0
        
        # yolo_labels[..., 3:5] /= num_grids
        return images, np.array(yolo_labels, dtype='float32')

    def on_epoch_end(self):
        # Shuffle the order every epoch while keeping image and label paths aligned
        if self.image_paths:
            combined = list(zip(self.image_paths, self.label_paths))
            np.random.shuffle(combined)
            self.image_paths, self.label_paths = zip(*combined)
        else:
            self.image_paths = []
            self.label_paths = []

class Result:
    def __init__(self, columns):
        self.columns = columns
    
      
if __name__ == '__main__':
    from plot_utils import plot_labels
    PATH_data = 'D:\\Year 3\\PyTorch tutorials\\StopSign\\data'
    IMAGE_DIR = PATH_data + '\\images'
    LABEL_DIR = PATH_data + '\\labels'
    IMAGE_SIZE = 224
    NUM_GRIDS = 7
    NUM_BBOXES = 2
    NUM_CLASSES = 4
    BATCH_SIZE = 32
    image_size = (IMAGE_SIZE, IMAGE_SIZE)

    val_loader = YOLOv1DataLoader(IMAGE_DIR, LABEL_DIR, image_size, 
                                num_grids=NUM_GRIDS, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE,
                                image_format='.png')
    
    for batch_idx, (images, labels) in enumerate(val_loader):
        print(batch_idx, images.shape, labels.shape)
        
        image = images[0]
        label = labels[0]
        plot_labels(image, yolo2xywh(label), num_grids=NUM_GRIDS)
        if batch_idx == 1:
            break