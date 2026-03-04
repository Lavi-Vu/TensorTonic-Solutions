import numpy as np
def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    stride = image_size / feature_size
    anchor = []
    for i in range (feature_size):
        for j in range (feature_size):
            center_x = (j + 0.5) * stride
            center_y = (i + 0.5) * stride
            for scale in scales:
                for aspect_ratio in aspect_ratios:
                    width = scale * np.sqrt(aspect_ratio)
                    height = scale / np.sqrt(aspect_ratio)
                    anchor.append([center_x - width/2, center_y - height/2, center_x + width/2, center_y + height/2])
    return anchor