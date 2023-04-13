
def MinMaxArray(fitnesses, vectors, minOrMax):
    ret = []

    goal_vectors, normalization_vectors = vectors

    for nv, gv in zip(normalization_vectors, goal_vectors):
        ret.append(MinMax(fitnesses, nv, gv, minOrMax))

    return tuple(ret)


def MinMax(fitnesses, normalization_vector, goal_vector, minOrMax):

    # If param count goal has been reached then push accuracy only
    count=0
    ret= (fitnesses[0] - goal_vector[0]) / normalization_vector[0] if minOrMax[0] ==0 else goal_vector[0]-fitnesses[0]/normalization_vector[0]
    if count < len(goal_vector)-1:
        ret= max(ret,
                 (fitnesses[count+1] - goal_vector[count+1]) / normalization_vector[count+1] if minOrMax[count+1] ==0 
                 else goal_vector[count+1]-fitnesses[count+1]/normalization_vector[count+1])
    return ret
