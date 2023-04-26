import math
###
# need to improve
####
from copy import deepcopy
class IndividualRecord:
    def __init__(self):

        self.gen_count = 0
        self.gens = []
        self.gens2 = []

    def add_gen(self, gen):
        self.gens.append([])
        for ind in gen:
            temp_array= deepcopy(ind.block_architecture.parameter)
            temp_array.extend(ind.fitness.values)
            self.gens [self.gen_count].append( 
                (temp_array)
            )
        self.gen_count += 1


    def save(self, gen_interval, test_name, title="Fig_None", comment=None):
        import matplotlib.pyplot as plt
        import math

        plot_rows = math.ceil(len(self.gens) / gen_interval / 2)
        fig, axes = plt.subplots(
            plot_rows, 2, sharey=True, sharex=True, figsize=(20, 10 * plot_rows)
        )

        if title:
            if comment:
                title = title + "_{}".format(comment)
            fig.suptitle(title)
        for i in range(0, self.gen_count, gen_interval):
            
            try:
                data=[]                                 #0 to n-2: data, n-1:  goal

                subplot_num = i // gen_interval
                sx = subplot_num // 2
                sy = subplot_num % 2
                j=0
                while j < len(self.gens[0][0]):
                    temp_array=[]
                    for single_data in self.gens[i]:
                        temp_array.append(single_data[j])
                    data.append(temp_array)
                    j+=1
                indicator=0
                while indicator < len(data)-2:
                    axes[sx, sy].scatter(data[0], data[indicator+1])
                    indicator += 1
                axes[sx, sy].set_title("Gen {}, count: {}".format(i, len(self.gens[i])))
                axes[sx, sy].set(xlabel="Param Count", ylabel="Accuracy")
            except Exception as e:
                pass

        from pathlib import Path

        path = "Output/{}/Figures".format(test_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig("Output/{}/Figures/{}".format(test_name, title))

    def goals(self, gen_interval, test_name):

        import matplotlib.pyplot as plt
        import math

        plot_cols = math.ceil(len(self.gens) / gen_interval / 2)
        fig, axes = plt.subplots(plot_cols, 2, sharey=True)
        fig.tight_layout(h_pad=2)
        fig.set_size_inches(20, 8 * plot_cols)

        goals = []
        data=[]                                 #0 to n-2: data, n-1:  goal

        from statistics import mean
        
        for i in range(0, self.gen_count, gen_interval):
            try:
                for single_data in self.gens[i]:
                    data.append(list(zip(single_data)))
                goals += [(i, g) for g in data[-1]]
            except Exception as e:
                pass

        import matplotlib.figure
        import matplotlib.backends.backend_agg as agg

        fig = matplotlib.figure.Figure(figsize=(45, 15))
        agg.FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)

        ax.title.set_text("Goals")

        ax.scatter(
            [ind[0] for ind in goals],
            [ind[1] for ind in goals],
            facecolor=(0.7, 0.7, 0.7),
            zorder=-1,
        )

        from pathlib import Path

        path = "Output/{}/Figures".format(test_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig("Output/{}/Figures/goals".format(test_name))

    def pareto(self, test_name):
        from Demos import get_global
        minOrMax= get_global("minOrMax")
        filter_funcs = get_global("filter_function_args")

        individuals = [list(ind) for ind in self.gens[-1]]              #last model
        minOrMax=get_global("minOrMax")
        best_models=[]
        
        i=0
        for param_count in list(set([ind[0] for ind in individuals])):
            best_models.append([param_count])
            flag=1
            for value in minOrMax:
                if flag==1:
                    flag=0
                    continue
                if value==0:
                    best_models[i].append(10000000000)
                else:
                    best_models[i].append(0)
            i+=1
        best_models.sort(key=lambda x: x[0])
        pcount=[]
        for ind in individuals:
            for models in best_models:
                pcount.append(models[0])
            bm_index=pcount.index(ind[0])
            #bm_index = [pcount for (pcount, acc) in best_models].index(ind[0])      #improve this
            i=1
            while i < len(best_models[0]):
                if minOrMax[i] == 1:
                    if ind[i] > best_models[bm_index][i]:
                        best_models[bm_index][i] = ind[i]
                else:
                    if ind[i] < best_models[bm_index][i]:
                        best_models[bm_index][i] = ind[i]
                
                i+=1

        pareto_inds = [best_models[0]]

        for ind_to_compare in best_models[1:]:

            is_dominated = False

            for existing_ind in pareto_inds:

                if a_dominates_b(existing_ind, ind_to_compare, minOrMax) or (
                    set(ind_to_compare) == set(existing_ind)
                ):

                    is_dominated = True
                    break

                elif a_dominates_b(ind_to_compare, existing_ind, minOrMax):
                    pareto_inds.remove(existing_ind)

            if is_dominated:
                continue
            else:
                pareto_inds.append(ind_to_compare)
        pareto=[]
        i=0
        while i< len(pareto_inds[0]):
            temp_pareto= [ind[i] for ind in pareto_inds]
            pareto.append(temp_pareto)
            i+=1
        pareto=list(pareto)
        import matplotlib.backends.backend_agg as agg
        import matplotlib.figure

        fig = matplotlib.figure.Figure(figsize=(45, 15))
        agg.FigureCanvasAgg(fig)
        ax=[]
        i=0
        graph_types = 3                            # 3 different graph types "Population", "Best for each Param Count", "pareto Front"
        
        pop_graph=1
        param_graph=2
        pareto_graph=3
        graph_colors=[[0.7,0.7,0.7],[0.3,0.7,0.7]]      #color of each point
        top_limit=[100,100000000]
        i=0
        while i<(len(ind)-2):
            ax = fig.add_subplot(2, 3, i*3+pop_graph)
            ax.title.set_text("Population")
            ax.set_ylim(bottom=0, top=100)
            ax.scatter(
                [ind[0] for ind in individuals],
                [ind[i+1] for ind in individuals],
                facecolor=(graph_colors[i][0],graph_colors[i][1],graph_colors[i][2]),
                zorder=-1,
            )
            for ff in filter_funcs[0]:
                m = -filter_funcs[1][0][i+1] / filter_funcs[1][0][0]
                c = ff[1] - (m * ff[0])
                y_point = [0, c]
                x_point = [-c / m, 0]
                ax.plot(ff[0], ff[1], "go")
                ax.plot(x_point, y_point)
            ax = fig.add_subplot(2, 3, i*3+param_graph)
            ax.set_xscale("log")
            ax.title.set_text("Best for Each Param Count")
            ax.set_ylim(bottom=0, top=top_limit[i])
            ax.scatter(
                [ind[0] for ind in best_models],
                [ind[i+1] for ind in best_models],
                facecolor=(graph_colors[i][0],graph_colors[i][1],graph_colors[i][2]),
                zorder=-1,
            )
            ax = fig.add_subplot(2, 3, i*3+pareto_graph)
            ax.plot(pareto[0],pareto[i+1])
            ax.scatter(pareto[0],pareto[i+1], facecolor=(graph_colors[i][0],graph_colors[i][1],graph_colors[i][2]), zorder=-1)
            ax.title.set_text("Pareto Front")
            ax.set_ylim(bottom=0, top=top_limit[i])   
            i+=1 

    
        # ax.set_xscale("log")
        

        from pathlib import Path

        path = "Output/{}/Figures".format(test_name)
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig("Output/{}/Figures/pareto".format(test_name))

        return pareto_inds


def a_dominates_b(a, b, minOrMax):
    n_better = 0

    # First index is parameter count, thus we want a[0] < b[0]
    index=0
    while index < min(len(a),len(b)):
        if minOrMax == 0:
            if a[index] < b[index]:
                n_better += 1
        index +=1

    # Second index is accuracy, thus we want a[1] > b[1]

    if n_better == min(len(a),len(b)):
        return True

    return False


def plot_hof_pareto(hof, test_name):
    import matplotlib

    x = [i.block_architecture.param_count for i in hof.items]
    y = [i.block_architecture.accuracy for i in hof.items]

    import matplotlib.backends.backend_agg as agg

    fig = matplotlib.figure.Figure(figsize=(15, 15))
    agg.FigureCanvasAgg(fig)

    padding = 1.1
    max_x = max(x) * padding
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(x, y, facecolor=(0.7, 0.7, 0.7), zorder=-1)
    ax.xscale = "log"

    for item in [(x[i], y[i]) for i in range(1, len(x))]:
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (item[0], 0),
                max_x - item[0],
                item[1],
                lw=0,
                facecolor=(1.0, 0.8, 0.8),
                zorder=-10,
            )
        )

    ax.set_xscale("log")
    ax.set_ylim(bottom=0, top=100)

    from pathlib import Path

    path = "Output/{}/Figures".format(test_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig("Output/{}/Figures/hof_pareto".format(test_name))
