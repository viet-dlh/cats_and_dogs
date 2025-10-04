import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd

DATASET_DIR = "/Users/hoangviet/Desktop/cats_and_dogs/data"   
EDA_RESULTS_DIR = "/Users/hoangviet/Desktop/cats_and_dogs/EDA_Results"

# tạo thư mục kết quả
os.makedirs(EDA_RESULTS_DIR, exist_ok=True)

def get_image_info(base_path):
    """
    Lấy thông tin ảnh từ dataset
    """
    data = []
    for label in ["Cat", "Dog"]:
        for split in ["train","validate", "test"]:
            folder = os.path.join(base_path, label, split)
            if not os.path.exists(folder):
                continue
            for file in os.listdir(folder):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    file_path = os.path.join(folder, file)
                    try:
                        with Image.open(file_path) as img:
                            width, height = img.size
                    except:
                        width, height = None, None
                    data.append({
                        "label": label,
                        "split": split,
                        "file": file,
                        "path": file_path,
                        "width": width,
                        "height": height
                    })
    return pd.DataFrame(data)

# Lấy thông tin ảnh
df = get_image_info(DATASET_DIR)

# Lưu summary ra file txt
summary_path = os.path.join(EDA_RESULTS_DIR, "summary.txt")
with open(summary_path, "w") as f:
    f.write("=== Dataset Summary ===\n")
    f.write(f"Tổng số ảnh: {len(df)}\n")
    f.write(f"Số lượng ảnh Mèo: {len(df[df['label']=='Cat'])}\n")
    f.write(f"Số lượng ảnh Chó: {len(df[df['label']=='Dog'])}\n")
    f.write("\n--- Train/Test split ---\n")
    f.write(str(df.groupby(["split", "label"]).size()))
    f.write("\n\n--- Kích thước ảnh (width x height) ---\n")
    f.write(str(df[["width","height"]].describe()))

print("Summary saved at:", summary_path)

# Vẽ phân phối train/test
plt.figure(figsize=(6,5))
sns.countplot(data=df, x="split", hue="label")
plt.title("Số lượng ảnh theo train/test")
plt.savefig(os.path.join(EDA_RESULTS_DIR, "train_test_distribution.png"))

# Vẽ phân phối label
plt.figure(figsize=(6,5))
sns.countplot(data=df, x="label")
plt.title("Số lượng ảnh theo label (Cat vs Dog)")
plt.savefig(os.path.join(EDA_RESULTS_DIR, "label_distribution.png"))

# Vẽ histogram độ phân giải
plt.figure(figsize=(8,5))
sns.histplot(df["width"].dropna(), bins=30, kde=True, color="blue", label="Width")
sns.histplot(df["height"].dropna(), bins=30, kde=True, color="orange", label="Height")
plt.legend()
plt.title("Phân phối kích thước ảnh (width & height)")
plt.savefig(os.path.join(EDA_RESULTS_DIR, "image_resolution_distribution.png"))

print("EDA plots saved in:", EDA_RESULTS_DIR)