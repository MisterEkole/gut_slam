import os
from PIL import Image

def convert_tiff_to_png(input_folder, output_folder):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

   
    tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.tif')]

    if not tiff_files:
        print("No TIFF files found in the input folder.")
        return

    for tiff_file in tiff_files:
        try:
           
            with Image.open(os.path.join(input_folder, tiff_file)) as img:
               
                png_file = os.path.splitext(tiff_file)[0] + '.png'
           
                img.save(os.path.join(output_folder, png_file), 'PNG')
                print(f"Converted {tiff_file} to {png_file}")
        except Exception as e:
            print(f"Failed to convert {tiff_file}: {e}")

if __name__ == "__main__":
    input_folder = "/Users/ekole/Dev/gut_slam/gut_images/Frames_S2000"
    output_folder = "/Users/ekole/Dev/gut_slam/gut_images/Frames_S2000_png"
    convert_tiff_to_png(input_folder, output_folder)
    print("Conversion complete.")
