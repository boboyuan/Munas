def get_config(args=None):
    ###
    # configure all 
    ###
    from Demos import set_global, get_config

    set_global("existing_generation", None)                             #initialize generations for genetic algorithm
    set_global("start_gen", 0)

    config = get_config(args)

    return config


def load_genetic_params_from_config(config):
    from Demos import set_global
    from TensorNAS.Tools.ConfigParse import (
        GetPopulationSize,
        GetGenerationCount,
        GetCrossoverProbability,
        GetVerboseMutation,
        GetMutationMethod,
        GetVariableMutationGenerationalChange,
        GetMutationAttempts,
        GetRetrainEveryGeneration,
        GetMutationProbability,
        GetSelfMutationProbability,
        GetTrainingSampleSize,
        GetTestSampleSize,
        GetGenerationGap,
        GetGenerationSaveInterval,
        GetUseReinforcementLearning,
        GetAlpha,
    )

    set_global("pop_size", GetPopulationSize(config))
    set_global("gen_count", GetGenerationCount(config))
    set_global("verbose_mutation", GetVerboseMutation(config))
    set_global("mutation_attempts", GetMutationAttempts(config))
    set_global("retrain_every_generation", GetRetrainEveryGeneration(config))
    mutation_method = GetMutationMethod(config)
    set_global("mutation_method", mutation_method)
    set_global(
        "variable_mutation_generational_change",
        GetVariableMutationGenerationalChange(config),
    )
    set_global("cxpb", GetCrossoverProbability(config))
    set_global("mutpb", GetMutationProbability(config))
    set_global("self_mutation_probability", GetSelfMutationProbability(config))
    set_global("training_sample_size", GetTrainingSampleSize(config))
    set_global("test_sample_size", GetTestSampleSize(config))
    set_global("generation_gap", GetGenerationGap(config))
    set_global("generation_save_interval", GetGenerationSaveInterval(config))
    set_global("use_reinforcement_learning", GetUseReinforcementLearning(config))
    set_global("alpha", GetAlpha(config))


def run_deap_test(generate_individual, evaluate_individual, crossover, mutate):
    from importlib import import_module
    from TensorNAS.Tools.DEAP.Test import setup_DEAP, register_DEAP_individual_gen_func
    from TensorNAS.Algorithms.EASimple import TestEASimple
    from Demos import get_global

    creator = import_module("deap.creator")
    toolbox = import_module("deap.base").Toolbox()

    setup_DEAP(
        creator=creator,
        toolbox=toolbox,
        objective_weights=get_global("weights"),
        multithreaded=get_global("multithreaded"),
        distributed=get_global("distributed"),
        thread_count=get_global("thread_count"),
    )

    register_DEAP_individual_gen_func(
        creator=creator, toolbox=toolbox, ind_gen_func=generate_individual
    )

    pop, logbook, test = TestEASimple(
        cxpb=get_global("cxpb"),
        mutpb=get_global("mutpb"),
        pop_size=get_global("pop_size"),
        gen_count=get_global("gen_count"),
        evaluate_individual=evaluate_individual,
        crossover_individual=crossover,
        mutate_individual=mutate,
        toolbox=toolbox,
        test_name=get_global("test_name"),
        verbose=get_global("verbose"),
        filter_function=get_global("filter_function"),
        filter_function_args=get_global("filter_function_args"),
        save_individuals=get_global("save_individuals"),
        generation_gap=get_global("generation_gap"),
        generation_save=get_global("generation_save_interval"),
        comment=get_global("comments"),
        multithreaded=get_global("multithreaded"),
        log=get_global("log"),
        start_gen=get_global("start_gen"),
    )

    return pop, logbook, test
