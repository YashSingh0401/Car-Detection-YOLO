import os

from ultralytics import YOLO
import cv2

# Load model
model = YOLO("models/yolov8n.pt")

# Vehicle class IDs (COCO dataset)
vehicle_classes = [2, 3, 5, 7]  # car, bike, bus, truck


# ================= IMAGE =================
import os
import cv2

def detect_all_images():
    input_folder = "input"
    output_folder = "output"

    # Create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_images = 0

    for file in os.listdir(input_folder):
        if file.endswith((".jpg", ".png", ".jpeg")):
            total_images += 1
            img_path = os.path.join(input_folder, file)

            print(f"\n📸 Processing: {file}")

            results = model(img_path)

            r = results[0]
            boxes = r.boxes

            vehicle_count = 0

            for box in boxes:
                cls = int(box.cls[0])

                if cls in [2, 3, 5, 7]:  # vehicles
                    vehicle_count += 1

            print(f"🚗 Vehicles detected: {vehicle_count}")

            # Save output image
            output_path = os.path.join(output_folder, file)
            annotated_img = r.plot()
            cv2.imwrite(output_path, annotated_img)

            print(f"💾 Saved to: {output_path}")

    if total_images == 0:
        print("❌ No images found in input folder")
    else:
        print(f"\n✅ Done! Processed {total_images} images")

    # Show result
    results[0].show()

    print("Detection completed!")


# ================= VIDEO =================
def detect_video():
    cap = cv2.VideoCapture("input/video.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame = results[0].plot()

        cv2.imshow("Video Detection", frame)

        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


# ================= WEBCAM =================
def detect_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        frame = results[0].plot()

        cv2.imshow("Webcam Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ================= MENU =================
print("1. Scan all images (Auto + Count + Save)")
print("2. Video Detection")
print("3. Webcam Detection")

choice = input("Enter choice: ")

if choice == "1":
    detect_all_images()

elif choice == "2":
    detect_video()

elif choice == "3":
    detect_webcam()

else:
    print("Invalid choice")
