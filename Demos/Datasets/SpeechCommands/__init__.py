import os

data_folder = "SpeechCommands"


def GetData(dataset_dir):
    from Demos.Datasets import tmp_dir

    tdir = tmp_dir

    from Demos.Datasets.SpeechCommands.kws_util import parse_command
    from Demos.Datasets.SpeechCommands.get_dataset import get_training_data

    if dataset_dir != "":
        tdir = dataset_dir

    data_dir = os.path.join(tdir, data_folder)

    Flags, unparsed = parse_command()
    Flags.data_dir = data_dir

    ds_train, ds_test, ds_val = get_training_data(Flags)

    train_len = 85511
    validation_len = 10102
    test_len = 4890

    # ds_train = ds_train.shuffle(train_len)
    # ds_test = ds_test.shuffle(test_len)
    # ds_val = ds_val.shuffle(validation_len)

    train_len = len(ds_train)
    validation_len = len(ds_val)
    test_len = len(ds_test)

    ds_train = ds_train.repeat()
    ds_test = ds_test.repeat()
    ds_val = ds_val.repeat()

    import tensorflow as tf

    shape = tuple(tf.compat.v1.data.get_output_shapes(ds_train)[0].as_list()[1:])

    return {
        "train_generator": ds_train,
        "train_len": train_len,
        "validation_generator": ds_val,
        "validation_len": validation_len,
        "test_generator": ds_test,
        "test_len": test_len,
        "input_tensor_shape": shape,
    }


def GetInputShape():

    return (49, 10, 1)
