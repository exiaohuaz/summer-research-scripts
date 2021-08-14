import os
import glob

f = "RGB7b22a986-f261-43c5-9961-65781796a1c5/" 
newdir = "rgb_pngs/"
def main():
  if not os.path.isdir(newdir):
    os.mkdir(newdir)
  for img in os.listdir(f):
    if img.endswith(".png"):
      print(img)
      os.rename(os.path.join(f, img), os.path.join(newdir, img))


if __name__ == "__main__":
  main()