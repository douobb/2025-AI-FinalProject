import os
import cv2
import numpy as np

input_root = "data/raw_images"
output_root1 = "data/processed_images"
output_root2 = "data/processed_images_no_epicenter"

os.makedirs(output_root1, exist_ok=True)
os.makedirs(output_root2, exist_ok=True)

for year_folder in os.listdir(input_root):
    print(f"processing - {year_folder}")
    year_input_path = os.path.join(input_root, year_folder)
    year_output_path1 = os.path.join(output_root1, year_folder)
    year_output_path2 = os.path.join(output_root2, year_folder)
    os.makedirs(year_output_path1, exist_ok=True)
    os.makedirs(year_output_path2, exist_ok=True)

    for filename in os.listdir(year_input_path):
        input_path = os.path.join(year_input_path, filename)
        output_path1 = os.path.join(year_output_path1, filename)
        output_path2 = os.path.join(year_output_path2, filename)
        basename = os.path.splitext(filename)[0]

        img = cv2.imread(input_path)

        # 處理
        if int(basename) < 2018111:
            if int(basename) >= 2009141 and int(basename) <= 2010128 and int(basename) != 2010108 and int(basename) != 2010124:
                # 去除文字
                if int(basename) == 2010110:
                    mask_img = cv2.imread('masks/471-600 mask.png')
                elif int(basename) == 2009144 or int(basename) == 2010009 or int(basename) == 2010015 or int(basename) == 2010043:
                    mask_img = cv2.imread('masks/480-600 mask.png')
                else:
                    mask_img = cv2.imread('masks/446-600 mask.png')
    
                target_color = [0, 0, 0]
                word_mask = np.all(mask_img == target_color, axis=-1).astype(np.uint8) * 255
                img = cv2.inpaint(img, word_mask, 3, cv2.INPAINT_TELEA)

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # 邊框
                c_green = [75, 255, 102]
                lower_green = np.array(c_green) - 40
                upper_green = np.array(c_green) + 40
                mask = cv2.inRange(hsv, lower_green, upper_green)
                black = np.full(img.shape, (0, 0, 0), dtype=np.uint8)
                img = np.where(mask[:, :, None] == 255, black, img)

                # 縣市界
                c_yellow = [30, 130, 153]
                lower_yellow = np.array(c_yellow) - 50
                upper_yellow = np.array(c_yellow) + 50
                mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                blue = np.full(img.shape, (255, 100, 100), dtype=np.uint8)
                img = np.where(mask[:, :, None] == 255, blue, img)

                # 斷層
                c_purple = [150, 80, 140]
                lower_purple = np.array(c_purple) - 50
                upper_purple = np.array(c_purple) + 50
                mask = cv2.inRange(hsv, lower_purple, upper_purple)
                green = np.full(img.shape, (50, 180, 50), dtype=np.uint8)
                img = np.where(mask[:, :, None] == 255, green, img)
            elif int(basename) == 2017001:
                mask_img = cv2.imread('masks/499-671 mask.png')

                # 去除文字
                target_color = [0, 0, 0]
                word_mask = np.all(mask_img == target_color, axis=-1).astype(np.uint8) * 255
                img = cv2.inpaint(img, word_mask, 3, cv2.INPAINT_TELEA)
                
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # 縣市界
                c_yellow = [30, 130, 153]
                lower_yellow = np.array(c_yellow) - 50
                upper_yellow = np.array(c_yellow) + 50
                mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                blue = np.full(img.shape, (255, 100, 100), dtype=np.uint8)
                img = np.where(mask[:, :, None] == 255, blue, img)

                # 斷層
                target_color = [4, 2, 5]
                mask = np.all(img == target_color, axis=-1).astype(np.uint8) * 255
                green = np.full(img.shape, (50, 180, 50), dtype=np.uint8)
                img = np.where(mask[:, :, None] == 255, green, img)

                target_color = [4, 2, 4]
                mask = np.all(img == target_color, axis=-1).astype(np.uint8) * 255
                green = np.full(img.shape, (50, 180, 50), dtype=np.uint8)
                img = np.where(mask[:, :, None] == 255, green, img)

                # 邊框
                c_green = [75, 255, 102]
                lower_green = np.array(c_green) - 10
                upper_green = np.array(c_green) + 10
                mask = cv2.inRange(hsv, lower_green, upper_green)
                black = np.full(img.shape, (0, 0, 0), dtype=np.uint8)
                img = np.where(mask[:, :, None] == 255, black, img)
            else:
                # 去除文字
                target_color = [0, 0, 0]
                word_mask = np.all(img == target_color, axis=-1).astype(np.uint8) * 255
                img = cv2.inpaint(img, word_mask, 3, cv2.INPAINT_TELEA)

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                # 邊框
                c_green = [75, 255, 102]
                lower_green = np.array(c_green) - 10
                upper_green = np.array(c_green) + 10
                mask = cv2.inRange(hsv, lower_green, upper_green)
                black = np.full(img.shape, (0, 0, 0), dtype=np.uint8)
                img = np.where(mask[:, :, None] == 255, black, img)

                # 縣市界
                c_yellow = [30, 170, 153]
                lower_yellow = np.array(c_yellow) - 10
                upper_yellow = np.array(c_yellow) + 10
                mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
                blue = np.full(img.shape, (255, 100, 100), dtype=np.uint8)
                img = np.where(mask[:, :, None] == 255, blue, img)

                # 斷層
                c_purple = [132, 98, 130]
                lower_purple = np.array(c_purple) - 10
                upper_purple = np.array(c_purple) + 10
                mask = cv2.inRange(hsv, lower_purple, upper_purple)
                green = np.full(img.shape, (50, 180, 50), dtype=np.uint8)
                img = np.where(mask[:, :, None] == 255, green, img)
        elif int(basename) < 2020001:
            if int(basename) == 2018131:
                    mask_img = cv2.imread('masks/499-638 mask.png')
            else:
                mask_img = cv2.imread('masks/499-671 mask.png')

            # 去除文字
            target_color = [0, 0, 0]
            word_mask = np.all(mask_img == target_color, axis=-1).astype(np.uint8) * 255
            img = cv2.inpaint(img, word_mask, 3, cv2.INPAINT_TELEA)
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # 縣市界
            c_yellow = [30, 130, 153]
            lower_yellow = np.array(c_yellow) - 50
            upper_yellow = np.array(c_yellow) + 50
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            blue = np.full(img.shape, (255, 100, 100), dtype=np.uint8)
            img = np.where(mask[:, :, None] == 255, blue, img)

            # 斷層
            target_color = [4, 2, 5]
            mask = np.all(img == target_color, axis=-1).astype(np.uint8) * 255
            green = np.full(img.shape, (50, 180, 50), dtype=np.uint8)
            img = np.where(mask[:, :, None] == 255, green, img)

            target_color = [4, 2, 4]
            mask = np.all(img == target_color, axis=-1).astype(np.uint8) * 255
            green = np.full(img.shape, (50, 180, 50), dtype=np.uint8)
            img = np.where(mask[:, :, None] == 255, green, img)

            # 邊框
            c_green = [75, 255, 102]
            lower_green = np.array(c_green) - 10
            upper_green = np.array(c_green) + 10
            mask = cv2.inRange(hsv, lower_green, upper_green)
            black = np.full(img.shape, (0, 0, 0), dtype=np.uint8)
            img = np.where(mask[:, :, None] == 255, black, img)
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            c_gray = [111, 59, 228]
            lower_gray = np.array(c_gray) - 40
            upper_gray = np.array(c_gray) + 40
            mask = cv2.inRange(hsv, lower_gray, upper_gray)
            color = np.full(img.shape, (255, 240, 230), dtype=np.uint8)
            img = np.where(mask[:, :, None] == 255, color, img)
            
            target_color = [255, 240, 230]
            mask = np.all(img == target_color, axis=-1).astype(np.uint8) * 255
            color = np.full(img.shape, (255, 255, 153), dtype=np.uint8)
            img = np.where(mask[:, :, None] == 255, color, img)
        
        # 裁切
        special_list = {2000026, 2000031, 2000091, 2000114, 2001057, 2002008, 2002026, 2002137, 2003057, 2003062, 2003090, 2003122, 2004021, 2004094, 2005148, 2006078, 2006088, 2006089, 2006106, 2006107, 2007058, 2008040, 2009102, 2009103, 2009125}
        h, w = img.shape[:2]
        if h == 2977 and w == 2334:
            img = img[400:2400, 600:1800]
        elif h == 600 and w == 427:
            if int(basename) in special_list:
                img = img[140:390, 160:310]
            else:
                img = img[100:425, 135:330]
        elif h == 600 and w == 446:
            img = img[100:425, 145:340]
        elif h == 600 and w == 480:
            img = img[140:390, 185:335]
        elif h == 600 and w == 471:
            img = img[135:385, 180:330]
        elif h == 671 and w == 499:
            img = img[95:470, 165:390]
        elif h == 638 and w == 499:
            img = img[140:390, 200:350]

        cv2.imwrite(output_path1, img)  # 儲存有震央的圖片

        # 清除震央
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([115, 170, 140])
        upper = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        if int(basename) >= 2020001:
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations = 15)
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        cv2.imwrite(output_path2, img)  # 儲存無震央的圖片

print("done")