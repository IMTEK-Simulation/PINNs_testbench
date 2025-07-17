from PINNLearning.data import conv_data
from IPython.display import clear_output, display
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import time


# This function creates an interactive 3D graph from the given data vectors
# corresponding to each axis
def int_3D_plot(X, Y, Z, title=None, xlabel='x', ylabel='y', zlabel='z'):
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
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel
        )
    )
    # remove the color bar
    fig.update_traces(showscale=False)

    return fig


# Plots a slideshow of the given graphs. For this purpose, the data must be
# presented as a list of 2D data sets.
def plot_animated(data, pause=0.001, iterator=1):
    # iterate though the complete data set
    for i, y in enumerate(data):
        clear_output(wait=True)

        # create the graphs
        fig, ax = plt.subplots()
        _ = ax.plot(y)  # Suppress output
        ax.set_title(f'Time Steps t={i * iterator}')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')

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


# Plot the error between a given PINN and a simulation in 1D or 2D
def plot_twoD_error(title, model, simulation, x_points, y_points=None,
                    xlabel='x', ylabel='y', elabel='Error magnitude'):

    # 1D case
    if y_points is None or np.ndim(simulation) == 1:
        # prepare the points and the dims for the heat map
        points = x_points.reshape(-1, 1)
        shape = (1, len(x_points))
        extent = [x_points[0], x_points[-1], 0, 1]
    # 2D case
    else:
        # prepare the 2D points for the PINN
        X, Y = np.meshgrid(x_points, y_points, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel()], axis=-1)
        shape = (x_points.shape[0], y_points.shape[0])
        # set the dims for the heat map
        extent = [x_points[0], x_points[-1], y_points[0], y_points[-1]]

    # inference the PINN on the data points of the simulation
    pinn_inp = conv_data(points)
    pinn_vals = model(pinn_inp).numpy().reshape(shape)
    error = np.abs(pinn_vals.T - simulation)

    # create the plot of the error in 1D and 2D
    fig, ax = plt.subplots(figsize=(8, 5))
    img = ax.imshow(error, cmap='hot', origin='lower', aspect='auto', extent=extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_points is None:
        ax.set_yticks([])  # 1D: suppress y labels
        ax.set_xticks(np.linspace(x_points[0], x_points[-1], min(6, len(x_points))))
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label(elabel)
    fig.tight_layout()

    return plt
