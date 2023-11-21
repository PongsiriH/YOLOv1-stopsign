import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

LABELS_LIST_STOPSIGN = ['trafficlight', 'speedlimit', 'crosswalk', 'stop']
LABELS_DICT_STOPSIGN = dict({0:'trafficlight', 1:'speedlimit', 2:'crosswalk', 3:'stop'})


def get_best_prediction(prediction):
    S = prediction.shape[0]
    confidences = np.array([prediction[..., :1], prediction[..., 5:6]])
    bboxes = np.array([prediction[..., 1:5], prediction[..., 6:10]])
    cls = np.array(prediction[..., 10:])
    
    # conf_idx = np.squeeze(np.argmax(confidences, axis=0), axis=-1)
    conf_idx = np.where(prediction[..., :1] > prediction[..., 5:6], 0, 1)

    best_conf = np.zeros((S, S, 1))
    best_bbox = np.zeros((S, S, 4))
    for i in range(S):
        for j in range(S):
            idx = conf_idx[i, j]
            best_conf[i, j, :] = confidences[idx, i, j, :]
            best_bbox[i, j, :] = bboxes[idx, i, j, :]
            
    return best_conf, best_bbox, cls

def apply_confidence_thresh(conf, conf_thresh=0.9):
    return np.where(conf >= conf_thresh, 1, 0)

def components2yolo(bbox, conf, cls):
    selected_bbox = bbox * conf
    selected_bbox = np.where(selected_bbox < 0, 0, selected_bbox)
    # print(selected_bbox.shape, conf.shape, cls.shape, np.concatenate([selected_bbox, cls], axis=-1).shape)
    return np.concatenate([conf, selected_bbox, cls], axis=-1)

def plot_labels(image, labels, class_labels=None, num_grids=None, relative_to_grids=True, 
                 title='', show_grid=True, grid_color='lightgray',
                ):
    """
    Plots bounding boxes on the image.

    :param image: The image as a numpy array.
    :param labels: List of tuples (class_id, x_center, y_center, width, height).
    :param class_labels: Optional dictionary mapping class IDs to class names.
    """
    if relative_to_grids and num_grids is None:
        raise ValueError('relative to grids but num_grids is not provided')
        
    plt.imshow(image)
    plt.title(title)
    img_height, img_width = image.shape[:2]

    if show_grid and num_grids is not None:
        plt.xticks([i*(img_width/num_grids) for i in range(num_grids)], minor=True)
        plt.yticks([i*(img_height/num_grids) for i in range(num_grids)], minor=True)
        plt.grid(which="minor", color=grid_color, linestyle='-', linewidth=0.5)

    for label in labels:
        class_id, x_center, y_center, width, height = label

        # Convert normalized coordinates to pixel values
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        if relative_to_grids:
            width /= num_grids
            height /= num_grids
        # Calculate the top-left corner of the rectangle
        x_min = x_center - width / 2
        y_min = y_center - height / 2

        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')

        # Add the rectangle to the Axes
        plt.gca().add_patch(rect)

        # Optionally add class labels
        if class_labels and class_id in class_labels:
            plt.text(x_min, y_min, class_labels[class_id], color='blue', fontsize=12)

