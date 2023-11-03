import argparse
from ocrtransform.model import detector


def pars_config():
    parser = argparse.ArgumentParser(description="a simple package to extarct the key informations from receipts images")
    parser.add_argument("--img_path", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = pars_config()
    det = detector.Detector()
    txt_res = det.forward_model(img_path=args.img_path)
    json_res = det.postprocess()
    print(json_res)