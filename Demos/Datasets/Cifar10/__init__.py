dataset_name = "Cifar10"
dataset_zip = "cifar-10-python.tar.gz"
dataset_url = "https://www.cs.toronto.edu/~kriz/{}".format(dataset_zip)


def GetData(dataset_dir):

    from Demos.Datasets import (
        make_dataset_dirs,
        tmp_dir,
        zip_dir,
        bar_progress,
    )
    import wget, os

    if dataset_dir != "":
        tmp_dir = dataset_dir
        zip_dir = dataset_dir + "/zips"

    make_dataset_dirs(dataset_name)

    if not os.path.isfile(os.path.join(zip_dir, dataset_zip)):
        print("Downloading Cifar10 dataset tar into: {}".format(zip_dir))
        wget.download(dataset_url, out=zip_dir, bar=bar_progress)
    else:
        print("Cifar10 tar already exists, skipping download")

    output_dir = os.path.join(tmp_dir, dataset_name)

    if not len(os.listdir(output_dir)):
        import tarfile

        tar_filepath = "{}/{}".format(zip_dir, dataset_zip)
        print("Extracting Cifar10 tar to: {}".format(tar_filepath))
        tar = tarfile.open(tar_filepath, "r:gz")
        tar.extractall(path=output_dir)
        tar.close()

        import shutil

        out_files = os.listdir(output_dir)

        sub_out_files = os.listdir(os.path.join(output_dir, out_files[0]))

        print(
            "Moving files {} into parent directory {}".format(sub_out_files, output_dir)
        )

        for file in sub_out_files:
            shutil.move(os.path.join(output_dir, out_files[0], file), output_dir)

        shutil.rmtree(os.path.join(output_dir, out_files[0]))
    else:
        print("Cifar10 tar already extracted")

    from Demos.Datasets.Cifar10.train import load_cifar_10_data

    (
        train_data,
        train_filenames,
        train_labels,
        test_data,
        test_filenames,
        test_labels,
        label_names,
    ) = load_cifar_10_data(output_dir)

    input_shape = GetInputShape()

    return {
        "test_data": test_data,
        "train_data": train_data,
        "test_labels": test_labels,
        "train_labels": train_labels,
        "input_tensor_shape": input_shape,
    }


def GetInputShape():

    return (32, 32, 3)
