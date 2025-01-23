import os

def calculate_face_height_width_percentage(face_coordinates):
    xmin, ymin, xmax, ymax = face_coordinates
    face_width = xmax - xmin
    face_height = ymax - ymin
    return (face_height / face_width) * 100

def process_all_annotations(training_dir):
    character_ratios = {}
    
    for subdir, _, files in os.walk(training_dir):
        for file_name in files:
            if not file_name.endswith(".txt"):
                continue
            
            file_path = os.path.join(subdir, file_name)
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    xmin, ymin, xmax, ymax = map(int, parts[1:5])
                    character = parts[5]
                    
                    height_width_percentage = calculate_face_height_width_percentage((xmin, ymin, xmax, ymax))
                    
                    if character not in character_ratios:
                        character_ratios[character] = []
                    character_ratios[character].append(height_width_percentage)
    
    for character, ratios in character_ratios.items():
        avg_percentage = sum(ratios) / len(ratios)
        print(f'{character} - Average Height/Width Percentage: {avg_percentage:.2f}%')

process_all_annotations('CAVA-2024-TEMA2/antrenare')
