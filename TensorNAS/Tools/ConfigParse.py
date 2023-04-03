vector_list=["Param", "Acc", "Flops"]           #change vector list for new value

n= []

minOrMax=[0,1,0]

def GetConfigFile(config_filename=None, directory=None):
    ###
    #Get config file from given directory and given config filename 
    ###
    import os, sys

    if directory:                                                           #if directory isn't empty
        script_path = directory                                             #set path to directory
    else:                                                                   #if directory parameter is empty
        script_path = os.path.abspath(os.path.dirname(sys.argv[0]))         #take the directory from the program directory

    import glob

    if config_filename:                                                     #if config_filename isn't empty
        # Relative filename
        config_file = glob.glob(
            script_path + "/{}".format(config_filename), recursive=False    #read file of a relative pathname of config_filname
        )

        # Absolute filename
        if len(config_file) == 0:                                           #if that file don't exist, then path must be absolute
            config_file = glob.glob(config_filename)
    else:
        config_file = glob.glob(script_path + "/**/*.cfg", recursive=False) #if no config_filename given, then use the first .cfg file in directory "script_path/**/"

    if len(config_file) == 0:                                               #if file is empty (e.g. no config file found)
        config_file = glob.glob(script_path + "/*.cfg", recursive=False)    #look for config file in "script_path/"

    if len(config_file) == 0:                                               #if still no file
        import TensorNAS                                                    

        tensornas_path = os.path.dirname(TensorNAS.__file__)                    
        config_file = glob.glob(
            tensornas_path + "/**/{}".format(config_filename), recursive=True   #look for configFile in TensorNAS directory
        )

    if len(config_file) == 0:
        raise Exception("Config file not found")                            #config file not found

    print("Config location: {}".format(config_file))

    return config_file


def LoadConfig(config_file):
    import configparser

    config = configparser.ConfigParser()
    config.read(config_file)                                                #read config file

    return config


def CopyConfig(config_filename, test_name):
    ###
    # Make Output directory and copy config file into it for better access
    ###
    from shutil import copyfile
    from pathlib import Path

    path = "Output/{}".format(test_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    copyfile(config_filename[-1], path + "/config.cfg")


def _GetFloat(parent, item):

    if not parent:
        return None

    if item not in parent:
        return 0.1

    return float(parent[item])


def _GetInt(parent, item, default=0):

    if not parent:
        return None

    if item not in parent:
        return default

    return int(parent[item])


def _GetBool(parent, item, default=False):

    if not parent:
        return default

    ret = parent.getboolean(item)

    if ret is None:
        return default

    return ret


def _GetStr(parent, item):

    if not parent:
        return None

    ret = str(parent[item])

    if ret != "None":
        return ret
    else:
        return None


def _GetGeneral(config):

    return config["general"]


def GetBlockArchitecture(config):

    return _GetGeneral(config)["BlockArchitecture"]


def GetClassCount(config):

    return int(_GetGeneral(config)["ClassCount"])


def GetVerbose(config):

    return _GetGeneral(config).getboolean("Verbose")


def GetMultithreaded(config):

    return _GetGeneral(config).getboolean("Multithreaded")


def GetDistributed(config):

    return _GetGeneral(config).getboolean("Distributed")


def GetDatasetModule(config):

    return _GetGeneral(config)["DatasetModule"]


def GetUseDatasetDirectory(config):

    try:
        return _GetStr(_GetGeneral(config), "DatasetDirectory")
    except Exception as e:
        return False

def GetDatasetDirectory(config):

    return _GetStr(_GetGeneral(config), "DatasetDirectory")

def GetLocalDataset(config):

    ret = _GetGeneral(config).getboolean("LocalDataset")
    if ret is None:
        ret = False
    return ret


def GetGenBlockArchitecture(config):

    return _GetGeneral(config)["GenBlockArchitecture"]


def GetThreadCount(config):

    return int(_GetGeneral(config)["ThreadCount"])


def GetGPU(config):

    return _GetGeneral(config).getboolean("GPU")


def GetLog(config):

    return _GetGeneral(config).getboolean("Log")


def _GetEvolution(config):

    return config["evolution"]


def GetCrossoverProbability(config):

    return float(_GetEvolution(config)["CrossoverProbability"])


def GetVerboseMutation(config):

    return _GetEvolution(config).getboolean("VerboseMutation")


def GetMutationMethod(config):

    return _GetStr(_GetEvolution(config), "MutationMethod")


def GetVariableMutationGenerationalChange(config):

    return float(_GetEvolution(config)["VariableMutationGenerationalChange"])


def GetMutationAttempts(config):

    return int(_GetEvolution(config)["MutationAttempts"])


def GetRetrainEveryGeneration(config):

    return _GetEvolution(config).getboolean("RetrainEveryGeneration")


def GetUseReinforcementLearning(config):

    return _GetBool(_GetEvolution(config), "UseReinforcementLearningMutation")


def GetAlpha(config):

    return float(_GetEvolution(config)["Alpha"])


def GetMutationProbability(config):

    try:
        return float(_GetEvolution(config)["MutationProbability"])
    except Exception:
        return 0.0


def GetSelfMutationProbability(config):

    try:
        return float(_GetEvolution(config)["SelfMutationProbability"])
    except Exception:
        return 0.2


def GetPopulationSize(config):

    return int(_GetEvolution(config)["PopulationSize"])


def GetGenerationCount(config):

    return int(_GetEvolution(config)["GenerationCount"])


def _GetOutput(config):

    return config["output"]


def GetGenerationGap(config):

    return int(_GetOutput(config)["GenerationGap"])


def GetGenerationSaveInterval(config):

    val = _GetOutput(config)["GenerationSave"]

    if val == "INTERVAL":
        return _GetGenerationSaveInterval(config)
    else:
        return 1


def _GetGenerationSaveInterval(config):

    return int(_GetOutput(config)["GenerationSaveInterval"])


def GetFigureTitle(config):

    return _GetOutput(config)["FigureTitle"]


def GetSaveIndividual(config):

    return _GetOutput(config).getboolean("SaveIndividuals")


def GetOutputPrefix(config):

    return _GetOutput(config)["OutputPrefix"]


def _GetGoals(config):

    return config["goals"]


def _GetFilters(config):

    return config["filter"]


def _GetVariableGoal(config):

    return _GetGoals(config).getboolean("VariableGoal")


def _GetNormalizationParamVectorStart(config):

    return int(_GetGoals(config)["NormalizationParamVectorStart"])


def _GetNormalizationParamVectorEnd(config):

    return int(_GetGoals(config)["NormalizationParamVectorEnd"])


def _GetNormalizationAccVectorStart(config):

    return float(_GetGoals(config)["NormalizationAccVectorStart"])


def _GetNormalizationAccVectorEnd(config):

    return float(_GetGoals(config)["NormalizationAccVectorEnd"])


def _GetNormalizationVectorSteps(config):

    return int(_GetGoals(config)["NormalizationVectorSteps"])


def _GetGoalVector(config):

    import ast

    return [n for n in ast.literal_eval(_GetGoals(config)["GoalVector"])]


def _GetGoalParamVectorStart(config):

    return int(_GetGoals(config)["GoalParamVectorStart"])


def _GetGoalParamVectorEnd(config):

    return int(_GetGoals(config)["GoalParamVectorEnd"])


def _GetGoalAccVectorStart(config):

    return int(_GetGoals(config)["GoalAccVectorStart"])


def _GetGoalAccVectorEnd(config):

    return int(_GetGoals(config)["GoalAccVectorEnd"])

############
# add additional goals below from here
def _GetFlopsParamVectorStart(config):                          

    return int(_GetGoals(config)["GoalFlopsVectorStart"])

def _GetFlopsParamVectorEnd(config):

    return int(_GetGoals(config)["GoalFlopsVectorEnd"])

#to here
###########

def _GetGoalVectorSteps(config):

    return int(_GetGoals(config)["GoalVectorSteps"])


def _GetNormalizationVector(config):

    import ast

    return [n for n in ast.literal_eval(_GetGoals(config)["NormalizationVector"])]


def GetFilterFunction(config):
    import importlib

    module = importlib.import_module(_GetFilterFunctionModule(config))
    func = getattr(module, _GetFilters(config)["FilterFunction"])

    return func


def GetUseGoalAttainment(config):

    return _GetBool(_GetFilters(config), "UseGoalAttainment", True)


def _GetFilterFunctionModule(config):

    return _GetFilters(config)["FilterFunctionModule"]


def GetWeights(config):

    config_arg = _GetFilters(config)["Weights"]

    if _GetVariableGoal(config):
        w_len = _GetGoalVectorSteps(config)
    else:
        w_len = _GetNormalizationVectorSteps(config)

    if config_arg == "minimize":
        return [-1] * w_len
    elif config_arg == "maximize":
        return [1] * w_len
    else:
        import ast

        return [n.strip() for n in ast.literal_eval(config_arg)]


def _GenVector(start, stop, steps):

    if steps > 1:

        if isinstance(start, float) or isinstance(stop, float):
            step = (stop - start) / (steps - 1)
        else:
            step = int((stop - start) / (steps - 1))

        if start == stop:
            return [start for i in range(steps)]
        else:
            if isinstance(start, float) or isinstance(stop, float):
                import numpy as np

                return [
                    i for i in np.arange(start, stop + step, 1 if step == 0 else step)
                ]
            else:
                return [i for i in range(start, stop + step, 1 if step == 0 else step)]

    else:

        return [
            (start + stop) / 2,
        ]


def _GenVariableVectors_old(p1_start, p1_stop, p2_start, p2_stop, p3_start, p3_stop, steps):

    v1 = _GenVector(p1_start, p1_stop, steps)
    v2 = _GenVector(p2_start, p2_stop, steps)
    v3 = _GenVector(p3_start, p3_stop, steps)

    return list(zip(v1, v2, v3))


def _GenVectorsVariableGoal_old(
    g_param_start, g_param_stop, g_acc_start, g_acc_stop, g_flops_start, g_flops_end,  steps, n1, n2, n3
):
    ###########
    #add the parameters for g_flops
    goal_vectors = _GenVariableVectors_old(
        g_param_start, g_param_stop, g_acc_start, g_acc_stop, g_flops_start, g_flops_end, steps
    )

    normalization_vectors = [(n1, n2 ,n3) for _ in range(len(goal_vectors))]

    return goal_vectors, normalization_vectors
    ########

def _GenVectorsVariableGoal(*arg):
    [g_start, g_end, steps, n] = arg
    ###########
    #add the parameters for g_flops
    goal_vectors = _GenVariableVectors(
        g_start, g_end, steps
    )
    normalization_vectors= [[]]
    normalization_vectors[0] = tuple(n)

    return goal_vectors, normalization_vectors
    ########


def _GenVectorsVariableNormilization(
    n_param_start, n_param_stop, n_acc_start, n_acc_stop, steps, g1, g2
):

    normalization_vectors = _GenVariableVectors(
        n_param_start, n_param_stop, n_acc_start, n_acc_stop, steps
    )

    goal_vectors = [(g1, g2) for _ in range(len(normalization_vectors))]

    return goal_vectors, normalization_vectors
def _GenVariableVectors(*args):
    v=[[]]
    [g_start, g_end, steps] = args
    for i in g_start:
        v[0].append(_GenVector(g_start[i],g_end[i],steps)[0])
    v[0]=tuple(v[0])
    return list(v)


def _GetGoalStartVector(config, vector):
    return int(_GetGoals(config)["Goal" +vector+ "VectorStart"])
def _GetGoalEndVector(config, vector):
    return int(_GetGoals(config)["Goal" +vector+ "VectorEnd"])


def GetFilterFunctionArgs(config):
    g_start={}
    g_end={}
    if _GetVariableGoal(config):
        # Goal vector varies, normilization vector is static
        for i in vector_list:
            g_start[i]=_GetGoalStartVector(config,i)
            g_end[i]=_GetGoalEndVector(config,i)
        n=_GetNormalizationVector(config)

        g_param_start = _GetGoalParamVectorStart(config)
        g_param_stop = _GetGoalParamVectorEnd(config)
        g_acc_start = _GetGoalAccVectorStart(config)
        g_acc_stop = _GetGoalAccVectorEnd(config)
        #############
        #additional goals here
        g_flops_start= _GetFlopsParamVectorStart(config)
        g_flops_end= _GetFlopsParamVectorEnd(config)
        #until here
        ###########
        steps = _GetGoalVectorSteps(config)
        #add another normalization value here
        n1, n2, n3 = _GetNormalizationVector(config)
        _GenVectorsVariableGoal_old(g_param_start,g_param_stop,g_acc_start,g_acc_stop,g_flops_start,g_flops_end,steps, n1, n2, n3)
        return _GenVectorsVariableGoal(
            g_start,g_end,steps,n
        )
    else:
        n_param_start = _GetNormalizationParamVectorStart(config)
        n_param_stop = _GetNormalizationParamVectorEnd(config)
        n_acc_start = _GetNormalizationAccVectorStart(config)
        n_acc_stop = _GetNormalizationAccVectorEnd(config)
        steps = _GetNormalizationVectorSteps(config)
        g1, g2 = _GetGoalVector(config)
        return _GenVectorsVariableNormilization(
            n_param_start, n_param_stop, n_acc_start, n_acc_stop, 0, 0, steps, g1, g2, 0 
        )


def _GetTensorflow(config):

    return config["tensorflow"]


def GetTrainingSampleSize(config):

    return _GetInt(_GetTensorflow(config), "TrainingSampleSize")


def GetTestSampleSize(config):

    return _GetInt(_GetTensorflow(config), "TestSampleSize")


def GetValidationSampleSize(config):

    return _GetInt(_GetTensorflow(config), "ValidationSampleSize")


def GetValidationSplit(config):

    return _GetFloat(_GetTensorflow(config), "ValidationSplit")


def GetTFOptimizer(config):

    return _GetTensorflow(config)["Optimizer"]


def GetTFLoss(config):

    return _GetTensorflow(config)["Loss"]


def GetTFMetrics(config):

    return _GetStr(_GetTensorflow(config), "Metrics")


def GetTFEarlyStopper(config):

    return _GetBool(_GetTensorflow(config), "EarlyStopper")


def GetTFPatience(config):

    return _GetInt(_GetTensorflow(config), "Patience")


def GetTFStopperMonitor(config):

    return _GetStr(_GetTensorflow(config), "StopperMonitor")


def GetTFStopperMinDelta(config):

    return _GetFloat(_GetTensorflow(config), "StopperMinDelta")


def GetTFStopperMode(config):

    return _GetStr(_GetTensorflow(config), "StopperMode")


def GetTFBatchSize(config):

    return _GetInt(_GetTensorflow(config), "BatchSize")


def GetTFTestBatchSize(config):

    return _GetInt(_GetTensorflow(config), "TestBatchSize")


def GetTFEpochs(config):

    return int(_GetTensorflow(config)["Epochs"])


def GetTFQuantizationAware(config):

    return _GetTensorflow(config).getboolean("QuantizationAware")


def GetTFUseClearMemory(config):

    return _GetBool(_GetTensorflow(config), "UseClearMemory")


def _GetLRScheduler(config):

    try:
        return config["lrscheduler"]
    except KeyError:
        return None


def GetUseLRScheduler(config):

    return _GetBool(_GetLRScheduler(config), "UseLRScheduler")


def GetLRScheduler(config):

    return _GetStr(_GetLRScheduler(config), "LRScheduler")


def GetLRInitialLearningRate(config):

    return _GetFloat(_GetLRScheduler(config), "InitialLearningRate")


def GetLRDecayPerEpoch(config):

    return _GetFloat(_GetLRScheduler(config), "DecayPerEpoch")


def _GetImageDataGeneartor(config):

    try:
        return config["image data generator"]
    except KeyError:
        return None


def UseImageDataGenerator(config):

    if _GetImageDataGeneartor(config):
        return True
    return False


def GetRotationRange(config):

    return _GetInt(_GetImageDataGeneartor(config), "RotationRange")


def GetWidthShiftRange(config):

    return _GetFloat(_GetImageDataGeneartor(config), "WidthShiftRange")


def GetHeightShiftRange(config):

    return _GetFloat(_GetImageDataGeneartor(config), "HeightShiftRange")


def GetHorizontalFlip(config):

    return _GetBool(_GetImageDataGeneartor(config), "HorizontalFlip")


def GetImageDataGeneratorValidationSplit(config):

    return _GetFloat(_GetImageDataGeneartor(config), "ValidationSplit")
