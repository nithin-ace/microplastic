import os
import cv2
import random
import shutil
from ultralytics import YOLO

# ==============================
# PATHS
# ==============================
DATASET_DIR = "dataset"
MASK_DIR = "segmented"
LABELS_DIR = "labels"

IMG_TRAIN = "images/train"
IMG_VAL = "images/val"
LBL_TRAIN = "labels/train"
LBL_VAL = "labels/val"

# ==============================
# STEP 1: LABEL GENERATION
# ==============================
def generate_labels():
    os.makedirs(LABELS_DIR, exist_ok=True)

    for mask_file in os.listdir(MASK_DIR):
        if not mask_file.endswith(".png"):
            continue

        mask_path = os.path.join(MASK_DIR, mask_file)

        image_name = mask_file.split("-")[0] + ".png"
        label_name = image_name.replace(".png", ".txt")

        mask = cv2.imread(mask_path)

        if mask is None:
            continue

        # convert to grayscale
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = mask.shape

        with open(os.path.join(LABELS_DIR, label_name), "w") as f:
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)

                if bw < 5 or bh < 5:
                    continue

                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                bw /= w
                bh /= h

                f.write(f"0 {xc} {yc} {bw} {bh}\n")

    print("✅ Labels ready")


# ==============================
# STEP 2: DATA SPLIT
# ==============================
def split_dataset():
    os.makedirs(IMG_TRAIN, exist_ok=True)
    os.makedirs(IMG_VAL, exist_ok=True)
    os.makedirs(LBL_TRAIN, exist_ok=True)
    os.makedirs(LBL_VAL, exist_ok=True)

    images = [f for f in os.listdir(DATASET_DIR) if f.endswith(".png") and "-1" not in f]

    random.shuffle(images)
    split = int(0.8 * len(images))

    train = images[:split]
    val = images[split:]

    def copy_files(files, img_dest, lbl_dest):
        for img in files:
            label = img.replace(".png", ".txt")

            img_path = os.path.join(DATASET_DIR, img)
            lbl_path = os.path.join(LABELS_DIR, label)

            if not os.path.exists(lbl_path):
                continue

            shutil.copy(img_path, os.path.join(img_dest, img))
            shutil.copy(lbl_path, os.path.join(lbl_dest, label))

    copy_files(train, IMG_TRAIN, LBL_TRAIN)
    copy_files(val, IMG_VAL, LBL_VAL)

    print("✅ Dataset ready")


# ==============================
# STEP 3: YAML
# ==============================
def create_yaml():
    with open("data.yaml", "w") as f:
        f.write(f"""
path: .
train: {IMG_TRAIN}
val: {IMG_VAL}

names:
  0: microplastic
""")
    print("✅ YAML ready")


# ==============================
# STEP 4: FAST TRAIN (🔥 OPTIMIZED)
# ==============================
def train_model():
    model = YOLO("yolov8n.pt")  # fastest model

    model.train(
        data="data.yaml",
        epochs=15,        # 🔥 FAST (10 min)
        imgsz=512,        # smaller = faster
        batch=8,
        patience=5,       # early stop
        workers=2,
        plots=False,
        save_period=-1,
        name="microplastic_fast"
    )

    print("✅ Training done")


# ==============================
# STEP 5: PREDICTION
# ==============================
def predict_image(image_path, volume_ml=10):
    model = YOLO("runs/detect/microplastic_fast/weights/best.pt")

    results = model(image_path)[0]

    count = len(results.boxes)
    concentration = (count / volume_ml) * 1000

    if concentration < 500:
        level = "Low"
    elif concentration < 2000:
        level = "Moderate"
    else:
        level = "High"

    print("\n🔍 RESULT")
    print("Particles:", count)
    print("Concentration:", round(concentration, 2))
    print("Level:", level)

    img = cv2.imread(image_path)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    print("\n🚀 Step 1")
    generate_labels()

    print("\n🚀 Step 2")
    split_dataset()

    print("\n🚀 Step 3")
    create_yaml()

    print("\n🚀 Step 4 (FAST TRAIN ~10 mins)")
    train_model()

    print("\n🎯 Testing...")

    test_image = os.path.join(DATASET_DIR, os.listdir(DATASET_DIR)[0])
    predict_image(test_image)
