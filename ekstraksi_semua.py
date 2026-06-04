import os
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

data = []

folders = [
    ("dataset/data/test/ripe", "ripe"),
    ("dataset/data/test/unripe", "unripe")
]

for folder, label in folders:

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        glcm = graycomatrix(
            gray,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )

        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        data.append([
            contrast,
            correlation,
            energy,
            homogeneity,
            label
        ])

df = pd.DataFrame(
    data,
    columns=[
        "Contrast",
        "Correlation",
        "Energy",
        "Homogeneity",
        "Label"
    ]
)

df.to_csv("hasil_test.csv", index=False)

print(df.head())
print("Data berhasil disimpan!")