import sys
from PIL import Image
import os


def micrify(in_file, out_file, scale=0.5):
    assert in_file != out_file
    img = Image.open(in_file)  # type: Image.Image
    w, h = img.size
    wc, hc = int(w / 2), int(h / 2)

    img = img.resize((wc, hc), Image.BICUBIC)
    img.save(out_file)
    img.close()


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('input_dir')
    p.add_argument('output_dir')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    if not args.overwrite:
        if os.path.exists(args.output_dir):
            print('directory {} is already existed'.format(args.output_dir), file=sys.stderr)
            sys.exit(1)
        os.makedirs(args.output_dir)

    for dirpath, _, files in os.walk(args.input_dir):
        for img in files:
            e = os.path.splitext(img)
            if e[1] not in ['.jpg', '.png', '.jpeg']:
                continue
            micrify(os.path.join(dirpath, img), os.path.join(args.output_dir, e[0] + '.png'))


if __name__ == '__main__':
    main()
