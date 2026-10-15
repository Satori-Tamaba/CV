import time

import cv2
from  pathlib import  Path


def show(title, img):
    cv2.imshow(title, img)

def save(output_dir, name, img):
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir/name), img)
    print(f"saved {output_dir / name }")

class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.perf_counter()
        duration = (end - self.start)
        print(f"[TIME] {self.name}, duration {duration:.4f}")


