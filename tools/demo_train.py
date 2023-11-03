import argparse
from ocrtransform.model import trainer


def pars_config():
    parser = argparse.ArgumentParser(description="a simple package to extarct the key informations from receipts images")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="summerization_model")
    parser.add_argument("--augment", type=bool, default=True)
    parser.add_argument("--imgs_path", type=str, default="dataset/images")
    parser.add_argument("--anno_path", type=str, default="dataset/key_information/key_information.json")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = pars_config()
    trainer_obj = trainer.Trainer(
        imgs_path=args.imgs_path, anno_path=args.anno_path,
        batch_size=args.batch_size, epochs=args.epochs, augment=args.augment,
        output_dir=args.out_dir
    )

    trainer_obj.train()
