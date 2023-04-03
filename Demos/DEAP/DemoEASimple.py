if __name__ == "__main__":

    import argparse
    import os, sys
    print(os.path.abspath(os.curdir))
    sys.path.append(os.path.abspath(os.curdir))
    from Demos import (                                             #import globals and init
        load_globals_from_config,
        load_tensorflow_params_from_config,
        set_test_train_data,
        get_global,
        set_global,
        evaluate_individual,
        mutate_individual,
    )
    from Demos.DEAP import (                                        #import genetic algorithm parameters
        load_genetic_params_from_config,
        run_deap_test,
        get_config,
    )
    from TensorNAS.Core.Crossover import crossover_individuals_sp

    parser = argparse.ArgumentParser()                              #import argument parser

    parser.add_argument(                                            #path to folder implemented with --folder (path to folder/string)
        "--folder",
        help="Absolute path to folder where interrupted test's output is stored",
        default=None,
    )
    parser.add_argument(                                            #generation of Genetic Algorithm where test resume (int)
        "--gen", help="Generation from which the test should resume", type=int
    )

    parser.add_argument(                                            #path to config file being used
        "--config",
        help="Location of config file to be used, default is to use first found config file in current working directory, then parent directories",
        type=str,
        default=None,
    )

    args = parser.parse_args()                                      #parse arguments

    config = get_config(args=args)

    load_globals_from_config(config)                                #load different parameters from config
    load_genetic_params_from_config(config)
    load_tensorflow_params_from_config(config)

    dataset_module = get_global("dataset_module")

    ## Multithreaded programs will want to get dataset data external to the parallelized evaluation step.
    ## Distributed progrems will want to get dataset data during the distributed evaliations steps such that they have
    ## local copies of the dataset.
    if get_global("multithreaded") or not get_global("local_dataset"):
        set_test_train_data(
            **dataset_module.GetData(get_global("dataset_directory")),
            training_sample_size=get_global("training_sample_size"),
            test_sample_size=get_global("test_sample_size"),
            validation_sample_size=get_global("validation_sample_size"),
            batch_size=get_global("batch_size"),
            validation_split=get_global("validation_split")
        )

    set_global("input_tensor_shape", dataset_module.GetInputShape())
    gba = get_global("gen_block_architecture")

    pop, logbook, test = run_deap_test(
        generate_individual=gba,
        evaluate_individual=evaluate_individual,
        crossover=crossover_individuals_sp,
        mutate=mutate_individual,
    )

    print("Done")
