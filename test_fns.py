from loss import compute_iou
import tensorflow as tf
import numpy as np

class TEST_compute_iou:
    def __init__(self, eps=1e-6):
        self.eps = 1e-6
        self.test_results = None
        
    def test(self):
        self.test_results = dict(
            test_overlapping = self.test_overlapping(),
            test_not_over_lapping = self.test_not_over_lapping()
        )
        return self.test_results
    
    def test_overlapping(self, print_results=False):
        # Define two sets of bounding boxes in 'xywh' format
        # shape=(2, 4) meaning 2 samples and 4 for [x,y,w,h].
        bbox1 = np.array([[100, 100, 50, 50], [200, 200, 60, 40]], dtype='float')
        bbox2 = np.array([[120, 120, 40, 60], [220, 220, 40, 60]], dtype='float')
        
        bbox1 = tf.convert_to_tensor(bbox1)
        bbox2 = tf.convert_to_tensor(bbox2)

        iou_values = compute_iou(bbox1, bbox2, mode='xywh').numpy()

        hand_calculated_1 = 25*35 / (40*60 + 50*50 - 25*35)
        hand_calculated_2 = 30*30 / (40*60 + 40*60 - 30*30)
        hand_calculated = [hand_calculated_1, hand_calculated_2]
        if print_results:
            print("IoU Values:", iou_values)
            print("Expected: ", hand_calculated)
            
        return np.allclose(iou_values, hand_calculated, self.eps)

    def test_not_over_lapping(self, print_results=False):
        bbox1 = np.array([0, 0, 50, 50], dtype='float')
        bbox2 = np.array([100, 100, 50, 50], dtype='float')

        bbox1 = tf.convert_to_tensor(bbox1)
        bbox2 = tf.convert_to_tensor(bbox2)
        
        iou_values = compute_iou(bbox1, bbox2, mode='xywh').numpy()             
        zero_not_overlapped = 0.0
        
        if print_results:
            print("IoU Values:", iou_values)
            print("Expected: ", [zero_not_overlapped])
        return np.allclose(iou_values, zero_not_overlapped, self.eps)

if __name__ == '__main__':
    test_compute_iou = TEST_compute_iou()
    print(test_compute_iou.test())
    