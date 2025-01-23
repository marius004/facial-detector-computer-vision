import matplotlib.pyplot as plt

def get_initial_window_sizes():
    return [(25, 30), (30, 25),
            (25, 18), (18, 25),
            (25, 27), (27, 25),
            (25, 20), (20, 25),
            (25, 25)]

def generate_scaled_window_sizes(height, width, scale_factor=1.3):
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

rectangles = calculate_sliding_window_sizes(480, 360)
print(len(rectangles))

fig, ax = plt.subplots(figsize=(10, 10))
for rect in rectangles:
    ax.add_patch(plt.Rectangle((0, 0), rect[0], rect[1], fill=False, edgecolor='blue'))
    
ax.set_xlim(0, 400)
ax.set_ylim(0, 400)
ax.set_aspect('equal', adjustable='box')
ax.set_title("Plot of Rectangles")

plt.show()
