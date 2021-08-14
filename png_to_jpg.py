from PIL import Image
import os

folder = "test_raw"

def makejpgs(f):
  for filename in os.listdir(f):
    im = Image.open(f + "/" + filename)
    rgb_im = im.convert('RGB')
    rgb_im.save(f + "/" + filename + ".jpg") #replace rgb_jpgs with rgb directory from dataset
    # avoids having to do extensive renaming while pulling filenames from json in tftranslate. only requires concatenation.
    print("saved " + filename)

def main():
  for f in os.listdir(folder):
    makejpgs(os.path.join(folder, f, "images"))

if __name__ == "__main__":
  main()