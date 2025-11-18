from pathlib import Path
import multiprocessing

import cv2
import dlib
import numpy as np
from PIL import Image

test_dataset_path = Path("./data/generated/generated")
output_dataset_path = Path("./data/generated/preprocessed")
output_dataset_path.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".avi", ".mp4"}

num_workers = min(max(1, multiprocessing.cpu_count() - 1), 8)
print(f"Using {num_workers} worker processes for preprocessing.")


def get_boundingbox(face, width, height):
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * 1.3)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


def detect_and_crop_face_optimized(image: Image.Image, resize_for_detection=640):
    if image.mode != "RGB":
        image = image.convert("RGB")
    original_np = np.array(image)
    original_h, original_w, _ = original_np.shape
    if original_w > resize_for_detection:
        scale = resize_for_detection / float(original_w)
        resized_h = int(original_h * scale)
        resized_np = cv2.resize(
            original_np, (resize_for_detection, resized_h), interpolation=cv2.INTER_AREA
        )
    else:
        scale = 1.0
        resized_np = original_np

    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(resized_np, 1)

    if not faces:
        return None
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    scaled_face_rect = dlib.rectangle(
        left=int(face.left() / scale),
        top=int(face.top() / scale),
        right=int(face.right() / scale),
        bottom=int(face.bottom() / scale),
    )
    x, y, size = get_boundingbox(scaled_face_rect, original_w, original_h)
    cropped_np = original_np[y : y + size, x : x + size]
    face_img = Image.fromarray(cropped_np)
    return face_img


def process_and_save_file(file_path):
    print(f"processing {file_path.name}")

    ext = file_path.suffix.lower()
    num_frames_to_extract = 30
    saved_files = []

    try:
        # Handle Images
        if ext in IMAGE_EXTS:
            image = Image.open(file_path)
            face_img = detect_and_crop_face_optimized(image)

            if face_img:
                # Save exactly as cropped (NO RESIZING)
                save_name = output_dataset_path / f"{file_path.stem}.jpg"
                face_img.save(save_name, quality=95)
                saved_files.append(save_name.name)

        # Handle Videos
        elif ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(file_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames > 0:
                frame_indices = np.linspace(
                    0, total_frames - 1, num_frames_to_extract, dtype=int
                )
                for i, idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    face_img = detect_and_crop_face_optimized(image)

                    if face_img:
                        # Save exactly as cropped (NO RESIZING)
                        save_name = (
                            output_dataset_path / f"{file_path.stem}_frame_{i:02d}.jpg"
                        )
                        face_img.save(save_name, quality=95)
                        saved_files.append(save_name.name)
            cap.release()

    except Exception as e:
        return file_path.name, f"Error: {str(e)}"

    return file_path.name, f"Saved {len(saved_files)} images"


if __name__ == "__main__":
    files = [p for p in sorted(test_dataset_path.iterdir()) if p.is_file()]
    print(f"Found {len(files)} files in {test_dataset_path}")

    with multiprocessing.Pool(processes=num_workers) as pool:
        for filename, status in pool.imap_unordered(process_and_save_file, files):
            print(f"{filename}: {status}")

    print(f"Processing finished. Output saved to {output_dataset_path}")
