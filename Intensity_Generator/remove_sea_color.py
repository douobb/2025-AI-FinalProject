from PIL import Image
import numpy as np
import os

input_root_1 = 'Final Project\Intensity_Generator\data_split\processed_images_no_epicenter\\train'
input_root_2 = 'Final Project\Intensity_Generator\data_split\processed_images_no_epicenter\\test'
output_root_1 = 'Final Project\Intensity_Generator\data_split\sea_removed\\train'
output_root_2 = 'Final Project\Intensity_Generator\data_split\sea_removed\\test'


def main():
    for filename in os.listdir(input_root_1):
        input_path = os.path.join(input_root_1, filename)
        output_path = os.path.join(output_root_1, filename)

        image = Image.open(input_path)
        image = image.convert("RGBA") # Ensure image is in RGBA format for transparency
        data = np.array(image)

        red, green, blue, alpha = data.T
        target_colors: tuple[int,int,int] = [
            (153,255,255),  # the sea  
            (154,255,255),  # the sea  
            (152,255,255),  # the sea  
            (151,255,255),  # the sea  
            (150,251,251),  # the sea  
            (150,252,251),  # the sea  
            (152,254,253),  # the sea  
            (152,253,253),  # the sea  
            (151,253,253),  # the sea  
            (148,249,247),  # the sea  
            (153,254,254),  # the sea  
            (154,254,254),  # the sea  
            (155,254,254),  # the sea  
            (151,252,252),  # the sea  
            (202,202,153),  # the sea  
            (164,177,110),  # the sea  
            (44,133,92),  # the sea  
            (63,162,131),  # the sea  
            (149,246,251),  # the sea  
            (141,243,240),  # the sea  
            (152,254,254),  # the sea  
            (144,245,242),  # the sea  
            (148,247,246),  # the sea  
            (84,81,221),  # the sea  
            (100,100,255),  # county boarder
            (199,199,255),  # county boarder
            (167,167,255),  # county boarder
            (102,102,255),  # county boarder
            (103,103,255),  # county boarder
            (106,106,255),  # county boarder
            (158,158,255),  # county boarder
            (94,94,240),  # county boarder
            (58,182,58),  # fault line
            (50,180,50),  # fault line
            (53,184,53),  # fault line
            (56,182,56),  # fault line
            (60,183,60),  # fault line
            (20,93,62),  # fault line
            (0,0,0),  # taiwan edge black line
            (23,23,23),  # taiwan edge black line
            (47,47,47),  # taiwan edge black line
            (87,87,87),  # taiwan edge black line
            (159,159,177),  # taiwan edge black line
            (203,203,203),  # taiwan edge black line
            (137,143,160),  # taiwan edge black line
            (72,142,112),  # taiwan edge black line
            (105,177,164),  # taiwan edge black line
            (182,183,188),  # taiwan edge black line
            (208,208,162),  # taiwan edge black line
            (107,182,149),  # taiwan edge black line
            (96,114,134),  # taiwan edge black line
            (75,167,135),  # taiwan edge black line
            (62,150,108),  # taiwan edge black line
            (224,224,224),  # mountains
            (151,157,165),  # mountains
            (220,220,220)  # mountains
            ]
        new_color = (255, 255, 255) # replace with White
        mask = False
        for color in target_colors:
            r, g, b = color
            mask |= (red == r) & (green == g) & (blue == b)

        data[..., :-1][mask.T] = new_color

        new_image = Image.fromarray(data)
        new_image.save(output_path)

    for filename in os.listdir(input_root_2):
        input_path = os.path.join(input_root_2, filename)
        output_path = os.path.join(output_root_2, filename)

        image = Image.open(input_path)
        image = image.convert("RGBA") # Ensure image is in RGBA format for transparency
        data = np.array(image)

        red, green, blue, alpha = data.T
        target_colors: tuple[int,int,int] = [
            (153,255,255),  # the sea  
            (154,255,255),  # the sea  
            (152,255,255),  # the sea  
            (151,255,255),  # the sea  
            (150,251,251),  # the sea  
            (150,252,251),  # the sea  
            (152,254,253),  # the sea  
            (152,253,253),  # the sea  
            (151,253,253),  # the sea  
            (148,249,247),  # the sea  
            (153,254,254),  # the sea  
            (154,254,254),  # the sea  
            (155,254,254),  # the sea  
            (151,252,252),  # the sea  
            (202,202,153),  # the sea  
            (164,177,110),  # the sea  
            (44,133,92),  # the sea  
            (63,162,131),  # the sea  
            (149,246,251),  # the sea  
            (141,243,240),  # the sea  
            (152,254,254),  # the sea  
            (144,245,242),  # the sea  
            (148,247,246),  # the sea  
            (84,81,221),  # the sea  
            (100,100,255),  # county boarder
            (199,199,255),  # county boarder
            (167,167,255),  # county boarder
            (102,102,255),  # county boarder
            (103,103,255),  # county boarder
            (106,106,255),  # county boarder
            (158,158,255),  # county boarder
            (94,94,240),  # county boarder
            (58,182,58),  # fault line
            (50,180,50),  # fault line
            (53,184,53),  # fault line
            (56,182,56),  # fault line
            (60,183,60),  # fault line
            (20,93,62),  # fault line
            (0,0,0),  # taiwan edge black line
            (23,23,23),  # taiwan edge black line
            (47,47,47),  # taiwan edge black line
            (87,87,87),  # taiwan edge black line
            (159,159,177),  # taiwan edge black line
            (203,203,203),  # taiwan edge black line
            (137,143,160),  # taiwan edge black line
            (72,142,112),  # taiwan edge black line
            (105,177,164),  # taiwan edge black line
            (182,183,188),  # taiwan edge black line
            (208,208,162),  # taiwan edge black line
            (107,182,149),  # taiwan edge black line
            (96,114,134),  # taiwan edge black line
            (75,167,135),  # taiwan edge black line
            (62,150,108),  # taiwan edge black line
            (224,224,224),  # mountains
            (151,157,165),  # mountains
            (220,220,220)  # mountains
            ]
        new_color = (255, 255, 255) # replace with White
        mask = False
        for color in target_colors:
            r, g, b = color
            mask |= (red == r) & (green == g) & (blue == b)

        data[..., :-1][mask.T] = new_color

        new_image = Image.fromarray(data)
        new_image.save(output_path)

if __name__ == '__main__':
    main()