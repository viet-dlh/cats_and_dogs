import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATASET_DIR = "/Users/hoangviet/Desktop/cats_and_dogs/data"   # đổi theo đường dẫn của bạn
EDA_RESULTS_DIR = "/Users/hoangviet/Desktop/cats_and_dogs/EDA_Results"

# Tạo thư mục kết quả
os.makedirs(EDA_RESULTS_DIR, exist_ok=True)

# Hàm lấy thông tin ảnh

def get_image_info(base_path):
    records = []

    for label in ["Cat", "Dog"]:
        label_path = os.path.join(base_path, label)
        for split in ["train", "validate", "test"]:
            split_path = os.path.join(label_path, split)
            if not os.path.exists(split_path):
                continue

            for filename in os.listdir(split_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(split_path, filename)
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            records.append({
                                "label": label,
                                "split": split,
                                "width": width,
                                "height": height,
                                "path": img_path
                            })
                    except Exception as e:
                        print(f"⚠️ Lỗi đọc ảnh {img_path}: {e}")

    return pd.DataFrame(records)



# Phân tích EDA

print("🔍 Đang đọc dữ liệu...")
df = get_image_info(DATASET_DIR)
print(f"Đã đọc {len(df)} ảnh")

# Thống kê tổng quan
summary = {
    "Tổng số ảnh": len(df),
    "Số lượng ảnh Mèo": (df["label"] == "Cat").sum(),
    "Số lượng ảnh Chó": (df["label"] == "Dog").sum(),
}
summary_df = pd.DataFrame(list(summary.items()), columns=["Thông tin", "Giá trị"])
summary_df.to_csv(os.path.join(EDA_RESULTS_DIR, "summary.csv"), index=False)
print(summary_df)

# Phân phối dữ liệu theo tập
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="split", hue="label")
plt.title("Phân phối ảnh theo tập và nhãn")
plt.xlabel("Tập dữ liệu")
plt.ylabel("Số lượng ảnh")
plt.tight_layout()
plt.savefig(os.path.join(EDA_RESULTS_DIR, "data_distribution.png"))
plt.close()

# Kích thước ảnh
df["aspect_ratio"] = df["width"] / df["height"]
plt.figure(figsize=(8, 5))
sns.histplot(df["aspect_ratio"], bins=30, kde=True)
plt.title("Phân phối tỉ lệ khung hình (width/height)")
plt.tight_layout()
plt.savefig(os.path.join(EDA_RESULTS_DIR, "aspect_ratio_distribution.png"))
plt.close()

# Độ phân giải (tổng pixel)
df["resolution"] = df["width"] * df["height"]
plt.figure(figsize=(8, 5))
sns.histplot(df["resolution"], bins=30, kde=True)
plt.title("Phân phối độ phân giải ảnh")
plt.tight_layout()
plt.savefig(os.path.join(EDA_RESULTS_DIR, "resolution_distribution.png"))
plt.close()

# Trung bình kích thước ảnh
size_summary = df.groupby("label")[["width", "height"]].mean().round(1)
size_summary.to_csv(os.path.join(EDA_RESULTS_DIR, "avg_image_size.csv"))


print("Đã lưu toàn bộ kết quả vào thư mục EDA_Results/")
