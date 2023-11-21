import tensorflow as tf

def compute_iou(bbox1, bbox2, mode='xywh', eps=1e-6):
    if mode == 'xywh':
        # Converting boxes from (x, y, w, h) to (xmin, ymin, xmax, ymax)
        bbox1_xyxy = tf.concat([bbox1[..., :2] - bbox1[..., 2:] / 2.0,
                                    bbox1[..., :2] + bbox1[..., 2:] / 2.0], axis=-1)
        bbox2_xyxy = tf.concat([bbox2[..., :2] - bbox2[..., 2:] / 2.0,
                                    bbox2[..., :2] + bbox2[..., 2:] / 2.0], axis=-1)
    elif mode == 'xyxy':
        bbox1_xyxy = bbox1
        bbox2_xyxy = bbox2 
    else:
        raise ValueError("Invalid mode. Expected 'xywh' or 'xyxy'.")    
    
    # Calculating the intersection areas
    intersect_mins = tf.maximum(bbox1_xyxy[..., :2], bbox2_xyxy[..., :2])
    intersect_maxes = tf.minimum(bbox1_xyxy[..., 2:], bbox2_xyxy[..., 2:])
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Calculating the union areas
    true_area = bbox1[..., 2] * bbox1[..., 3]
    pred_area = bbox2[..., 2] * bbox2[..., 3]
    union_area = true_area + pred_area - intersect_area

    # Computing the IoU
    iou = intersect_area / (union_area + eps)

    return iou

class YOLOv1Loss(tf.keras.losses.Loss):
    def __init__(
            self, num_grids, num_classes, num_bboxes=2, 
            lambda_coord=5, lambda_obj=1, lambda_noobj=0.5, lambda_cls=1, 
            eps=1e-6
                 ):
        self.S = num_grids
        self.C = num_classes
        self.B = num_bboxes        
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj    
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        self.eps = eps
    
    def __call__(self, y_true, y_pred):
        return self.call(y_true, y_pred)
    
    def call(self, y_true, y_pred):
        iou_box1 = tf.stop_gradient(compute_iou(y_true[..., 1:5], y_pred[..., 1:5], 'xywh', self.eps))
        iou_box2 = tf.stop_gradient(compute_iou(y_true[..., 1:5], y_pred[..., 6:10], 'xywh', self.eps))
        responsible_mask = tf.stop_gradient(tf.expand_dims(tf.where(iou_box1 > iou_box2, 1.0, 0.0), axis=-1))
        
        true_obj = y_true[..., :1]
        true_bbox_sqrt = true_obj * tf.concat([y_true[..., 1:3], tf.sqrt(y_true[..., 3:5]+self.eps)], axis=-1)
        true_cls = true_obj * y_true[..., 5:]
        
        pred_obj = true_obj * (
            responsible_mask * y_pred[..., :1]
            + (1-responsible_mask) * y_pred[..., 5:6]
            ) 
        pred_bbox = true_obj * (
            responsible_mask * y_pred[..., 1:5]
            + (1-responsible_mask) * y_pred[..., 6:10]
            ) 

        pred_bbox_sqrt = tf.concat([pred_bbox[..., :2], tf.sign(pred_bbox[..., 2:]) * tf.sqrt(tf.abs(pred_bbox[..., 2:])+self.eps)], axis=-1)
        pred_cls = y_pred[..., 10:]
        
        bbox_loss = true_obj * tf.square(pred_bbox_sqrt - true_bbox_sqrt)
        obj_loss = true_obj * tf.square(pred_obj - true_obj) 
        noobj_loss = (1-true_obj) * (
            tf.square(y_pred[..., :1] - true_obj)
            + tf.square(y_pred[..., 5:6] - true_obj)
        )
        cls_loss = true_obj * (
            tf.square(pred_cls - true_cls)
        )
        
        total_loss = tf.reduce_sum(
            self.lambda_coord * tf.reduce_sum(bbox_loss, axis=[1,2,3])
            + self.lambda_obj * tf.reduce_sum(obj_loss, axis=[1,2,3])
            + self.lambda_noobj * tf.reduce_sum(noobj_loss, axis=[1,2,3])
            + self.lambda_cls * tf.reduce_sum(cls_loss, axis=[1,2,3])
        )
        
        components_loss = {
            'bbox_loss' : tf.reduce_mean(bbox_loss),
            'obj_loss' : tf.reduce_mean(obj_loss),
            'noobj_loss' : tf.reduce_mean(noobj_loss),
            'cls_loss' : tf.reduce_mean(cls_loss),
        }
        
        return total_loss, components_loss