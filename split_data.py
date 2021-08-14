import os
import shutil

SRC = 'real/train'
DST = 'real_split'

#NUM_TRAIN = 8000
#NUM_VAL = 1889
#NUM_TEST = 0


def main():
    if not os.path.isdir(DST):
        os.mkdir(DST)

    train_dir = os.path.join(DST, 'train')
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)

    val_dir = os.path.join(DST, 'val')
    if not os.path.isdir(val_dir):
        os.mkdir(val_dir)

    """test_dir = os.path.join(DST, 'test')
    os.mkdir(test_dir)"""

    for class_label in os.listdir(SRC):
        train_dir_class = os.path.join(train_dir, class_label)
        os.mkdir(train_dir_class)

        src_dir_class = os.path.join(SRC, class_label)
        dirs = os.listdir(src_dir_class)
        image_paths = iter(sorted(dirs))

        numtrain = int(len(dirs) * .8)
        numval = int(len(dirs) * .2)

        for i in range(numtrain):
            src_path = os.path.join(src_dir_class, next(image_paths))
            shutil.copy(src_path, train_dir_class)

        val_dir_class = os.path.join(val_dir, class_label)
        os.mkdir(val_dir_class)

        for i in range(numval):
            src_path = os.path.join(src_dir_class, next(image_paths))
            shutil.copy(src_path, val_dir_class)



if __name__ == '__main__':
    main()