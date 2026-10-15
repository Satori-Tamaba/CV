import  argparse, sys, time
from pathlib import  Path
import  cv2, numpy as np
from utils import show, Timer, save
from kernel import GaussianKernel


def process(image_path, ksize, sigma, mode, output_dir, viz):
    src = cv2.imread(str(image_path))
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    if src is None:
        sys.exit(f"Cannot read {image_path}")
    results = {}
    if mode in ("custom", "both"):
        g_ker = GaussianKernel(sigma, ksize)
        with Timer("Custom"):
            results["custom"] = g_ker(src)
    if mode in ("cv2", "both"):
        with Timer("cv2"):
            results["cv2"] = cv2.GaussianBlur(src, (ksize, ksize), sigma)
    if mode == "both":
        diff = cv2.absdiff(results["custom"], results["cv2"])
        results["diff"] = cv2.normalize(diff, None, 0,255, cv2.NORM_MINMAX)
        print(f"Max diff: {diff.max()}")


    if output_dir:
        save(output_dir, "pic_opencv.png", cv2.cvtColor(results["cv2"], cv2.COLOR_RGB2BGR))
    if viz:
        for title, img in results.items():
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_to_show = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_to_show = img

            show(title, img_to_show)

        cv2.waitKey(0)
        cv2.destroyAllWindows()




def build_parser():
    p = argparse.ArgumentParser(formatter_class= argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-i", "--image", type = Path, required=True)
    p.add_argument("-k", "--ksize", type = int, default=3)
    p.add_argument("-s", "--sigma", type = float, default=1)
    p.add_argument("-m", "--mode", choices=["custom", "cv2", "both"], default="both")
    p.add_argument("-o", "--output_dir", type = Path, default= None)
    p.add_argument("-v", "--viz", type = bool, default=True)

    return p



def main(argv = None):
    args = build_parser().parse_args(argv)
    if not args.image.exists():
        sys.exit(f"Not founrd {args.image}")

    process(args.image, args.ksize, args.sigma, args.mode, args.output_dir, args.viz)



if __name__ == "__main__":
    main()
