import cv2
import os
import numpy as np
from ultralytics import YOLO
import argparse
from collections import Counter

def detect_objects(image_path, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    
    print(f"Reading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Oops! Couldn't read the image at {image_path}")
        return
    
    height, width = image.shape[:2]
    
    text_height = 150
    text_area = np.zeros((text_height, width, 3), dtype=np.uint8)
    
    print("Detecting objects...")
    results = model(image)
    
    detected_objects = {}
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            if class_name in detected_objects:
                detected_objects[class_name].append(confidence)
            else:
                detected_objects[class_name] = [confidence]
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    y_offset = 30
    cv2.putText(text_area, "Objects Found in This Image:", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    y_offset += 40
    for obj_name, confidences in detected_objects.items():
        count = len(confidences)
        avg_confidence = sum(confidences) / len(confidences)
        text = f"• {obj_name}: {count} found (confidence: {avg_confidence:.2f})"
        cv2.putText(text_area, text, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
    
    combined_image = np.vstack((text_area, image))
    
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, combined_image)
    print(f"\nGreat! I've saved the result to: {output_path}")
    
    print("\nHere's what I found:")
    print("-" * 30)
    for obj_name, confidences in detected_objects.items():
        avg_confidence = sum(confidences) / len(confidences)
        count = len(confidences)
        print(f"• {obj_name}: {count} of these (confidence: {avg_confidence:.2f})")
    print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description='My Object Detection Tool')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to your image file')
    parser.add_argument('--output', type=str, default='output', 
                       help='Output folder for results (default: output)')
    
    args = parser.parse_args()
    
    if os.path.exists(args.image):
        detect_objects(args.image, args.output)
    else:
        print(f"Sorry, I couldn't find the image at {args.image}")
        print("Please make sure the path is correct!")
        print("Example: python object_detection.py --image path/to/your/image.jpg")

if __name__ == "__main__":
    main() 