import math
import tensorflow as tf


def get_config(args=None):
    from time import gmtime, strftime
    from TensorNAS.Tools.ConfigParse import (
        LoadConfig,
        GetConfigFile,
        GetBlockArchitecture,
    )
    from TensorNAS.Tools.ConfigParse import (
        GetOutputPrefix,
        CopyConfig,
    )

    globals()["test_name"] = None
    config = None                                                               #init config as none
    config_filename = "example"                                                 #init filename config

    if args:
        if args.folder:                                                         #if folder argument is not empty (param --folder in args)
            from pathlib import Path

            globals()["test_name"] = Path(args.folder).name                     #setting test name to the name of the file
            globals()["existing_generation"] = args.folder + "/Models/{}".format(       #reading existing generations from Models of the folder
                args.gen
            )
            globals()["start_gen"] = args.gen                                   #set the starting generation
            config_loc = GetConfigFile(directory=args.folder)                   #get config location from folder argument
            config = LoadConfig(config_loc)                                     #load the config on location cofig_log
        if args.config:                                                         #if config argument is not empty (param --config in args); Overwrites --folder param partly
            config_filename = args.config                                       #set filename is config argument
            config_loc = GetConfigFile(config_filename=args.config)             #get config file from config argument location
            config = LoadConfig(config_loc)                                     #load config
    else:
        config_loc = GetConfigFile(config_filename=config_filename)
        config = LoadConfig(config_loc)

    globals()["ba_name"] = GetBlockArchitecture(config)                         #get block architecture from config

    if not get_global("test_name"):
        test_name_prefix = GetOutputPrefix(config)
        set_global("test_name", strftime("%d_%m_%Y-%H_%M", gmtime()))
        if test_name_prefix:
            set_global("test_name", test_name_prefix + "_" + get_global("test_name"))
        set_global("test_name", get_global("test_name") + "_" + get_global("ba_name"))
        CopyConfig(config_loc, get_global("test_name"))

    return config


def gen_classification_ba():
    global ba_mod, input_tensor_shape, class_count, batch_size, test_batch_size, optimizer

    ba = None
    while ba is None:
        try:
            ba = ba_mod.Block(
                input_shape=input_tensor_shape,
                batch_size=batch_size,
                test_batch_size=test_batch_size,
                optimizer=optimizer,
                class_count=class_count,
            )
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            pass

    return ba


def gen_auc_ba():
    global ba_mod, input_tensor_shape, batch_size, optimizer

    try:
        ba = ba_mod.Block(
            input_shape=input_tensor_shape,
            batch_size=batch_size,
            optimizer=optimizer,
        )
        return ba
    except Exception as e:
        raise e


def evaluate_individual(individual, test_name, gen, logger, minOrMax, indices):
    global epochs, batch_size, loss, metrics, train_generator, validation_generator, use_clear_memory
    global test_generator, save_individuals, q_aware, steps_per_epoch, batch_size, test_batch_size, test_len
    global dataset_module, verbose, train_len, test_len, validation_split, validation_len

    if not get_global("multithreaded"):
        if not any(
            k in globals()
            for k in ("train_generator", "train_len" "test_generator", "test_len")
        ):
            from Demos import set_test_train_data

            set_test_train_data(
                **dataset_module.GetData(get_global("dataset_directory")),
                validation_split=validation_split,
                training_sample_size=get_global("training_sample_size"),
                test_sample_size=get_global("test_sample_size"),
                batch_size=batch_size,
            )
    parameters=[]
    parameters.extend( individual.evaluate(
        train_generator=train_generator,
        train_len=train_len,
        test_generator=test_generator,
        test_len=test_len,
        validation_generator=validation_generator,
        validation_len=validation_len,
        epochs=epochs,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        loss=loss,
        metrics=metrics,
        test_name=test_name,
        model_name="{}/{}".format(gen, individual.index),
        q_aware=q_aware,
        use_clear_memory=use_clear_memory,
        logger=logger,
        verbose=verbose,
        minOrMax=minOrMax,
        indices= indices
    ))

    return parameters


def mutate_individual(individual):
    from copy import deepcopy

    verbose = get_global("verbose_mutation")
    mutation_attempts = get_global("mutation_attempts")
    loss = get_global("loss")
    metrics = get_global("metrics")

    attempts = 0
    mutated = False
    mutation_operation, mutation_note, mutation_table_references = None, None, None

    while attempts < mutation_attempts and mutated == False:
        try:
            attempt = deepcopy(individual.block_architecture)
            
            (
                mutation_operation,
                mutation_note,
                mutation_table_references,
            ) = attempt.mutate(
                mutation_method=get_global("mutation_method"),
                mutation_probability=get_global("self_mutation_probability"),
                mutate_with_reinforcement_learning=get_global(
                    "use_reinforcement_learning"
                ),
                goal_attainment=get_global("use_goal_attainment"),
                verbose=verbose,
            )

            model = attempt.get_keras_model(loss=loss, metrics=metrics)
            if model == None:
                raise Exception("Getting mutated model failed")
            mutated = True
        except Exception as e:
            import traceback

            traceback.print_exc()
            if verbose:
                print("Mutation attempt #{} failed:".format(attempts + 1, e))
            pass
        attempts += 1

    if mutated:
        if verbose:
            print("Mutated successfully")
        individual.block_architecture = attempt
        from TensorNAS.Core.BlockArchitecture import Mutation

        individual.block_architecture.mutations.append(
            Mutation(
                mutation_table_references=mutation_table_references,
                mutation_function=mutation_operation,
                mutation_note=mutation_note,
            )
        )

    return (individual,)


def load_globals_from_config(config):
    from TensorNAS.Tools.ConfigParse import (
        GetBlockArchitecture,
        GetClassCount,
        GetLog,
        GetVerbose,
        GetMultithreaded,
        GetDistributed,
        GetDatasetModule,
        GetUseDatasetDirectory,
        GetDatasetDirectory,
        GetLocalDataset,
        GetGenBlockArchitecture,
        GetThreadCount,
        GetGPU,
        GetSaveIndividual,
        GetFilterFunction,
        GetFilterFunctionArgs,
        GetUseGoalAttainment,
        GetWeights,
        GetFigureTitle,
    )
    from TensorNAS.Tools.JSONImportExport import GetBlockMod

    set_global("minOrMax", [0,1])                                     #0= minimize 1= maximize #[0,1,0]
    set_global("vectorList", ["Param", "Acc"])                        #change vector list for new value #["Param","Acc", "Flops"]
    if(len(get_global("minOrMax")) != len(get_global("vectorList"))):
        raise SystemExit("Error: vectorList isn't the same length with minOrMax, which need to be to keep the program running")
    globals()["ba_name"] = GetBlockArchitecture(config)                     #dupe?
    globals()["class_count"] = GetClassCount(config)
    globals()["ba_mod"] = GetBlockMod(globals()["ba_name"])
    globals()["log"] = GetLog(config)
    globals()["verbose"] = GetVerbose(config)
    globals()["multithreaded"] = GetMultithreaded(config)
    globals()["distributed"] = GetDistributed(config)
    dm = GetDatasetModule(config)
    globals()["dataset_directory"] = ""
    if GetUseDatasetDirectory(config):
        globals()["dataset_directory"] = GetDatasetDirectory(config)
    components = dm.split(".")
    dm = __import__(dm)
    for comp in components[1:]:
        dm = getattr(dm, comp)
    globals()["dataset_module"] = dm
    globals()["local_dataset"] = GetLocalDataset(config)
    gba = GetGenBlockArchitecture(config)
    components = gba.split(".")
    # fund = components[-1]
    module = ".".join(components[:-1])
    gba = __import__(module)
    for comp in components[1:-1]:
        gba = getattr(gba, comp)
    globals()["gen_block_architecture"] = eval("gba.{}".format(components[-1]))
    globals()["thread_count"] = GetThreadCount(config)
    globals()["use_gpu"] = GetGPU(config)
    globals()["save_individuals"] = GetSaveIndividual(config)
    globals()["filter_function"] = GetFilterFunction(config)
    globals()["filter_function_args"] = GetFilterFunctionArgs(config)
    globals()["use_goal_attainment"] = GetUseGoalAttainment(config)
    globals()["weights"] = GetWeights(config)
    globals()["comments"] = GetFigureTitle(config)

    if globals()["use_gpu"]:
        from TensorNAS.Tools.TensorFlow import GPU as GPU

        GPU.config_GPU()
    else:
        # import os
        #
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            import tensorflow as tf

            # Disable all GPUS
            tf.config.set_visible_devices([], "GPU")
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != "GPU"
        except Exception as e:
            raise e

    if not globals()["verbose"]:
        import os
        import tensorflow as tf

        print("Suppressing verbosity")
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL


class DataGenerator(tf.keras.utils.Sequence):
    """
    See https://towardsdatascience.com/writing-custom-keras-generators-fe815d992c5a
    https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
    https://stackoverflow.com/questions/62916904/failed-copying-input-tensor-from-cpu-to-gpu-in-order-to-run-gatherve-dst-tensor
    """
    ###
    # dataset is too big and being seperated to baches
    ###
    def __init__(self, x_set, y_set, batch_size=1):

        assert len(x_set) == len(
            y_set
        ), "Arrays passed to DataGenerator have different lengths"

        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, item):

        batch_x = self.x[item * self.batch_size : (item + 1) * self.batch_size]
        batch_y = self.y[item * self.batch_size : (item + 1) * self.batch_size]

        return batch_x, batch_y


def _convert_array_to_datagen(array_x, array_y, batch_size=1):

    return DataGenerator(array_x, array_y, batch_size)


def set_test_train_data(
    train_data=None,
    train_labels=None,
    test_data=None,
    test_labels=None,
    train_generator=None,
    train_len=None,
    test_generator=None,
    test_len=None,
    validation_generator=None,
    validation_len=None,
    validation_split=None,
    batch_size=1,
    input_tensor_shape=None,
    training_sample_size=None,
    test_sample_size=None,
    validation_sample_size=None,
    **kwargs
):
    # TensorNAS only accepts DataGenerators, thus if data is provided as arrays then they must be
    # converted to custom DataGenerators that can hold whatever format the data is,
    # ie. potentially not images. Training using a DataGenerator does not support a validation_split thus
    # if a DataGenerator is to be created from a training data array it must also produce a val_generator
    # if one is not already been provided.
    if all(
        [
            (train_data is not None),
            (train_labels is not None),
            (test_data is not None),
            (test_labels is not None),
        ]
    ):
        train_len = len(train_data)
        test_len = len(test_data)

        # Set required dataset lengths
        if training_sample_size is not None:
            if training_sample_size > 0:
                if training_sample_size > train_len:
                    training_sample_size = train_len
                train_len = training_sample_size

        train_data = train_data[:train_len]
        train_labels = train_labels[:train_len]

        if test_sample_size is not None:
            if test_sample_size > 0:
                if test_sample_size > test_len:
                    test_sample_size = test_len
                test_len = test_sample_size

        # Cut datasets down to size
        test_data = test_data[:test_len]
        test_labels = test_labels[:test_len]

        # Split training data for validation data
        train_len = math.floor(len(train_data) * (1 - validation_split))
        test_len = len(test_data)
        val_len = math.floor(len(train_data) * validation_split)

        if validation_sample_size is not None:
            if validation_sample_size > 0:
                if validation_sample_size > val_len:
                    validation_sample_size = val_len
                val_len = validation_sample_size

        # Create validation generator

        if batch_size > val_len:
            vbatch_size = val_len
        else:
            vbatch_size = batch_size
        globals()["validation_generator"] = DataGenerator(
            x_set=train_data[train_len:],
            y_set=train_labels[train_len:],
            batch_size=vbatch_size,
        )
        globals()["validation_len"] = val_len

        # Resize training dataset now that validation data has been removed and used
        train_data = train_data[:train_len]
        train_labels = train_labels[:train_len]

        globals()["train_generator"] = DataGenerator(
            x_set=train_data[:train_len],
            y_set=train_labels[:train_len],
            batch_size=batch_size,
        )
        globals()["test_generator"] = DataGenerator(
            x_set=test_data, y_set=test_labels, batch_size=1
        )
        globals()["train_len"] = train_len
        globals()["test_len"] = test_len

    else:
        # Set required dataset lengths
        if training_sample_size is not None:
            if training_sample_size > 0:
                if training_sample_size < train_len:
                    train_len = training_sample_size

        if test_sample_size is not None:
            if test_sample_size > 0:
                if test_sample_size < test_len:
                    test_len = test_sample_size

        if validation_sample_size is not None:
            if validation_sample_size > 0:
                if validation_sample_size < validation_len:
                    validation_len = validation_sample_size

        globals()["train_generator"] = train_generator
        globals()["train_len"] = train_len
        globals()["test_generator"] = test_generator
        globals()["test_len"] = test_len
        globals()["validation_generator"] = validation_generator
        globals()["validation_len"] = validation_len

    globals()["input_tensor_shape"] = input_tensor_shape


def load_tensorflow_params_from_config(config):
    from TensorNAS.Tools.ConfigParse import (
        GetTFEpochs,
        GetTFBatchSize,
        GetTFTestBatchSize,
        GetTFOptimizer,
        GetTFLoss,
        GetTFMetrics,
        GetTFQuantizationAware,
        GetTFUseClearMemory,
        GetTrainingSampleSize,
        GetTestSampleSize,
        GetValidationSampleSize,
        GetValidationSplit,
        GetTFEarlyStopper,
        GetTFPatience,
        GetTFStopperMonitor,
        GetTFStopperMinDelta,
        GetTFStopperMode,
        GetUseLRScheduler,
        GetLRScheduler,
        GetLRInitialLearningRate,
        GetLRDecayPerEpoch,
        UseImageDataGenerator,
        GetRotationRange,
        GetWidthShiftRange,
        GetHeightShiftRange,
        GetHorizontalFlip,
        GetImageDataGeneratorValidationSplit,
    )

    globals()["epochs"] = GetTFEpochs(config)
    globals()["batch_size"] = GetTFBatchSize(config)
    globals()["test_batch_size"] = GetTFTestBatchSize(config)
    globals()["optimizer"] = GetTFOptimizer(config)
    globals()["loss"] = GetTFLoss(config)
    globals()["metrics"] = GetTFMetrics(config)
    globals()["q_aware"] = GetTFQuantizationAware(config)
    globals()["use_clear_memory"] = GetTFUseClearMemory(config)
    globals()["training_sample_size"] = GetTrainingSampleSize(config)
    globals()["test_sample_size"] = GetTestSampleSize(config)
    globals()["validation_sample_size"] = GetValidationSampleSize(config)
    globals()["validation_split"] = GetValidationSplit(config)
    globals()["early_stopper"] = GetTFEarlyStopper(config)
    if globals()["early_stopper"]:
        globals()["patience"] = GetTFPatience(config)
        globals()["stopper_monitor"] = GetTFStopperMonitor(config)
        globals()["stopper_min_delta"] = GetTFStopperMinDelta(config)
        globals()["stopper_mode"] = GetTFStopperMode(config)
    globals()["use_lrscheduler"] = GetUseLRScheduler(config)
    if globals()["use_lrscheduler"]:
        globals()["lrscheduler"] = GetLRScheduler(config)
        globals()["initial_learning_rate"] = GetLRInitialLearningRate(config)
        globals()["decay_per_epoch"] = GetLRDecayPerEpoch(config)

    globals()["use_image_data_generator"] = UseImageDataGenerator(config)
    if globals()["use_image_data_generator"]:
        globals()["rotation_range"] = GetRotationRange(config)
        globals()["width_shift_range"] = GetWidthShiftRange(config)
        globals()["height_shift_range"] = GetHeightShiftRange(config)
        globals()["horizontal_flip"] = GetHorizontalFlip(config)
        globals()[
            "image_data_generator_validation_split"
        ] = GetImageDataGeneratorValidationSplit(config)


def get_global(var_name):
    try:
        return globals()[var_name]
    except:
        return None


def set_global(var_name, val):

    globals()[var_name] = val
