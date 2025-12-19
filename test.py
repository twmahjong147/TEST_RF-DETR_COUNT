import os
import supervision as sv
from inference import get_model
from PIL import Image
from collections import Counter

SAMPLES_DIR = "samples"
OUTPUT_DIR = "annotated_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = get_model("rfdetr-large")

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF'}


# Collect all class names

for filename in os.listdir(SAMPLES_DIR):
    if not any(filename.endswith(ext) for ext in image_extensions):
        continue

    all_labels = []
    class_to_detections = {}

    image_path = os.path.join(SAMPLES_DIR, filename)
    image = Image.open(image_path)
    predictions = model.infer(image, confidence=0.001)[0]
    detections = sv.Detections.from_inference(predictions)
    labels = [prediction.class_name for prediction in predictions.predictions]
    all_labels.extend(labels)

    # Save normal annotated image
    annotated_image = image.copy()
    annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
    annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections, labels)
    sv.plot_image(annotated_image)
    output_path = os.path.join(OUTPUT_DIR, f"annotated_{filename}")
    annotated_image.save(output_path)
    print(f"Annotated image saved to {output_path}")

    # Generate and save per-class annotated images
    for class_name in set(labels):
        # Filter detections and labels for this class
        indices = [i for i, l in enumerate(labels) if l == class_name]
        if not indices:
            continue
        # Filter detections for this class
        class_detections = detections[indices]
        class_labels = [labels[i] for i in indices]
        class_annotated_image = image.copy()
        class_annotated_image = sv.BoxAnnotator(
            color=sv.ColorPalette.ROBOFLOW
        ).annotate(class_annotated_image, class_detections)
        class_annotated_image = sv.LabelAnnotator(
            color=sv.ColorPalette.ROBOFLOW
        ).annotate(class_annotated_image, class_detections, class_labels)
        # Create subdirectory for each image (without extension)
        filename_no_ext = os.path.splitext(filename)[0]
        class_dir = os.path.join(OUTPUT_DIR, filename_no_ext)
        os.makedirs(class_dir, exist_ok=True)
        class_output_path = os.path.join(
            class_dir, f"annotated_{class_name}_{filename}"
        )
        class_annotated_image.save(class_output_path)
        print(f"Annotated image for class '{class_name}' saved to {class_output_path}")

    # Group by class name and print count in descending order
    class_counts = Counter(all_labels)
    print("\nClass counts (descending):")
    output_lines = ["Class counts (descending):\n"]
    for class_name, count in class_counts.most_common():
        line = f"{class_name}: {count}"
        print(line)
        output_lines.append(line + "\n")
    # Save to file
    filename_no_ext = os.path.splitext(filename)[0]
    counts_output_path = os.path.join(OUTPUT_DIR, filename_no_ext, "class_counts.txt")
    with open(counts_output_path, "w") as f:
        f.writelines(output_lines)
    print(f"Class counts saved to {counts_output_path}")

