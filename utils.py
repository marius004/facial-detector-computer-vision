def calculate_iou(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)
    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou

def get_initial_window_sizes():
    return [(25, 30), (30, 25),
            (25, 18), (18, 25),
            (25, 27), (27, 25),
            (25, 20), (20, 25),
            (25, 25)]

def generate_scaled_window_sizes(height, width, scale_factor):
    max_height = int(height * 0.75)
    max_width = int(width * 0.75)
    
    min_window_sizes = get_initial_window_sizes()
    window_sizes = set()

    for min_window in min_window_sizes:
        current_window_w, current_window_h = min_window
        while current_window_w <= max_width and current_window_h <= max_height:
            window_sizes.add((current_window_h, current_window_w))
            current_window_w = min(int(current_window_w * scale_factor), width)
            current_window_h = min(int(current_window_h * scale_factor), height)

    return window_sizes

def calculate_sliding_window_sizes(height, width, scale_factor=1.4):
    window_sizes = generate_scaled_window_sizes(height, width, scale_factor)
    
    max_height = int(height * 0.95)
    max_width = int(width * 0.95)

    for width_percentage in range(95, 69, -5):
        curr_width = int(width * (width_percentage / 100))
        for scale in [1.1830, 1.0220, 0.7968, 0.7014, 1.0708]:
            curr_height = int(curr_width * scale)
            window_sizes.add((curr_width, curr_height))
            window_sizes.add((curr_height, curr_width))

    window_sizes = {ws for ws in window_sizes if ws[0] <= max_height and ws[1] <= max_width}
    return list(window_sizes)
