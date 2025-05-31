import requests
from PIL import Image, ImageFile
import cv2
from io import BytesIO
import os
import time

def main():
    image_count = 0
    size_dict = {}
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    error_log = open("error - not exist.txt", "w", encoding="utf-8")
    # 各年度有感地震數量
    year_n = {2025:94, 2024:514, 2023:85, 2022:184, 2021:113,
                   2020:74, 2019:66, 2018:139, 2017:60, 2016:112,
                   2015:100, 2014:154, 2013:166, 2012:214, 2011:172,
                   2010:153, 2009:154, 2008:102, 2007:91, 2006:110,
                   2005:166, 2004:113, 2003:148, 2002:196, 2001:136, 2000:266}
    base_dir = "data/raw_images"
    os.makedirs(base_dir, exist_ok=True)
    for i in range(2000, 2026):
        print(f"downloading - {i}")
        save_dir = os.path.join(base_dir, str(i))
        os.makedirs(save_dir, exist_ok=True)
        for j in range(1, year_n[i] + 1):
            time.sleep(0.01)
            url = f"https://scweb.cwa.gov.tw/webdata/drawTrace/plotContour/{i}/{i}{str(j).zfill(3)}"
            if i >= 2020: url += "a.png"
            else: url += ".gif"
            response = requests.get(url)
            
            if "image" not in response.headers.get("Content-Type", ""):
                print(f"{i}/{i}{str(j).zfill(3)} : not exist")
                error_log.write(f"{i}{str(j).zfill(3)}\n")
                continue

            filename = f"{i}{str(j).zfill(3)}.png"
            filepath = os.path.join(save_dir, filename)
            if i >= 2020:
                with open(filepath, "wb") as f:
                    f.write(response.content)
            else:
                image = Image.open(BytesIO(response.content))
                image.save(filepath, format="PNG")
            
            img = cv2.imread(filepath)
            h, w = img.shape[:2]
            key = (w, h)
            if key not in size_dict:
                size_dict[key] = []
            size_dict[key].append(f"{i}{str(j).zfill(3)}")
            image_count += 1
    print(f"done, download {image_count} images")
    with open('image size.txt', 'w', encoding='utf-8') as f:
        for size, files in size_dict.items():
            f.write(f"size {size[0]}x{size[1]}：\n")
            for fname in files:
                f.write(f"  - {fname}\n")
            f.write("\n")
if __name__ == '__main__':
    main()
