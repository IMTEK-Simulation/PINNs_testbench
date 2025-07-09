from IPython.display import clear_output, display
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import time


# This function creates an interactive 3D graph from the given data vectors
# corresponding to each axis
def int_3D_plot(X, Y, Z, title=None):
    # create a plotly interactive figure, it is created as a surface plot
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    # change the layout of the plot
    fig.update_layout(
        # set the dimensions of the final image
        width=600,
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        # adjust the title of the image
        title={
            'text': title,
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        # set the labels for the axis
        scene=dict(
            xaxis_title='x',
            yaxis_title='Time t',
            zaxis_title='u(x)'
        )
    )
    # remove the color bar
    fig.update_traces(showscale=False)

    return fig


# Plots a slideshow of the given graphs. For this purpose, the data must be
# presented as a list of 2D data sets.
def plot_animated(data, pause=0.1):
    # iterate though the complete data set
    for i, y in enumerate(data):
        clear_output(wait=True)

        # create the graphs
        fig, ax = plt.subplots()
        _ = ax.plot(y)  # Suppress output
        ax.set_title(f'Time Steps {i}')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')

        # immediately display the graphs
        display(fig)
        plt.close(fig)

        # take a pause between graphs
        time.sleep(pause)


# Plot a normal graph based on a list of the plots
def simp_plot(data, title, xlabel='x', ylabel='y'):
    # ensure that data is an proper iterable
    if not (isinstance(data, list) and isinstance(data[0], list)):
        data = [data]

    # create the figure
    plt.figure(figsize=(5, 3))
    # plot all items in the data set
    for item in data:
        # split the arguments into positonal arguments and arguments
        # with a keyword given as tuples of ("keyword", "value").
        positional = []
        keywords = {}
        for i in item:
            if isinstance(i, tuple) and len(i) == 2 and isinstance(i[0], str):
                keywords[i[0]] = i[1]
            else:
                positional.append(i)

        # check if item is a scatter plot by checking for the string 'scatter'
        if any(isinstance(i, str) and i == 'scatter' for i in positional):
            # remvove 'scatter'
            new = [i for i in positional if not (isinstance(i, str) and i == 'scatter')]
            plt.scatter(*new, **keywords)
        else:
            plt.plot(*positional, **keywords)

    # configure the graph
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(data) > 1:
        plt.legend()
    plt.title(title)

    return plt


# Plotting function for a single graph with y axis scaled in log10
def plot_err_vs_epoch(data, title, xlabel='x', ylabel='y'):
    # ensure that data is a single list
    if isinstance(data, list) and isinstance(data[0], list):
        raise ValueError

    # create the plot based on the provided data
    plt = simp_plot(data, title, xlabel, ylabel)

    # modify the plot to change the y axis
    y_order = np.floor(np.log10(np.min(data[1])))
    plt.yscale('log')
    plt.ylim(10**y_order, 1)

    return plt
