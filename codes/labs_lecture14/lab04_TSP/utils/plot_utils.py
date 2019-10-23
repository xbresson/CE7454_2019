import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from utils.graph_utils import *


def plot_tsp(plt, x_coord, W_val, W_target, title="default"):
    """
    Helper function to plot TSP tours.

    Args:
        plt: Matplotlib figure/subplot
        x_coord: Coordinates of nodes
        W_val: Edge values (distance) matrix
        W_target: One-hot matrix with 1s on groundtruth/predicted edges
        title: Title of figure/subplot

    Returns:
        plt: Updated figure/subplot

    """

    def _edges_to_node_pairs(W):
        """Helper function to convert edge matrix into pairs of adjacent nodes.
        """
        pairs = []
        for r in range(len(W)):
            for c in range(len(W)):
                if W[r][c] == 1:
                    pairs.append((r, c))
        return pairs

    G = nx.from_numpy_matrix(W_val)
    pos = dict(zip(range(len(x_coord)), x_coord.tolist()))
    node_pairs = _edges_to_node_pairs(W_target)
    colors = ['g'] + ['b'] * (len(x_coord) - 1)  # Green for 0th node, blue for others
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=50)
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=node_pairs, alpha=1, width=1, edge_color='r')
    plt.set_title(title)
    return plt


def plot_predictions(x_nodes_coord, x_edges_values, y_edges, y_pred_edges, num_plots=3):
    """
    Plots groundtruth TSP tour vs. predicted tours (without beamsearch).

    Args:
        x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)
        x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
        y_edges: Groundtruth labels for edges (batch_size, num_nodes, num_nodes)
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        num_plots: Number of figures to plot

    """
    y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
    y = y.argmax(dim=3)  # B x V x V
    for f_idx, idx in enumerate(np.random.choice(len(y), num_plots, replace=False)): 
        f = plt.figure(f_idx, figsize=(10, 5))
        x_coord = x_nodes_coord[idx].cpu().numpy()
        W_val = x_edges_values[idx].cpu().numpy()
        W_target = y_edges[idx].cpu().numpy()
        W_sol = y[idx].cpu().numpy()
        plt1 = f.add_subplot(121)
        plot_tsp(plt1, x_coord, W_val, W_target, 'Groundtruth: {:.3f}'.format(W_to_tour_len(W_target, W_val)))
        plt2 = f.add_subplot(122)
        plot_tsp(plt2, x_coord, W_val, W_sol, 'Prediction: {:.3f}'.format(W_to_tour_len(W_sol, W_val)))
        plt.show()


def plot_predictions_beamsearch(x_nodes_coord, x_edges_values, y_edges, y_pred_edges, bs_nodes, num_plots=3):
    """
    Plots groundtruth TSP tour vs. predicted tours (with beamsearch).

    Args:
        x_nodes_coord: Input node coordinates (batch_size, num_nodes, node_dim)
        x_edges_values: Input edge distance matrix (batch_size, num_nodes, num_nodes)
        y_edges: Groundtruth labels for edges (batch_size, num_nodes, num_nodes)
        y_pred_edges: Predictions for edges (batch_size, num_nodes, num_nodes)
        bs_nodes: Predicted node ordering in TSP tours after beamsearch (batch_size, num_nodes)
        num_plots: Number of figures to plot

    """
    y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
    y = y.argmax(dim=3)  # B x V x V
    for f_idx, idx in enumerate(np.random.choice(len(y), num_plots, replace=False)):
    #for f_idx, idx in enumerate(np.random.choice(len(y), num_plots, replace=True)):
        f = plt.figure(f_idx, figsize=(15, 5))
        x_coord = x_nodes_coord[idx].cpu().numpy()
        W_val = x_edges_values[idx].cpu().numpy()
        W_target = y_edges[idx].cpu().numpy()
        W_sol = y[idx].cpu().numpy()
        W_bs = tour_nodes_to_W(bs_nodes[idx].cpu().numpy())
        plt1 = f.add_subplot(131)
        plot_tsp(plt1, x_coord, W_val, W_target, 'Groundtruth: {:.3f}'.format(W_to_tour_len(W_target, W_val)))
        plt2 = f.add_subplot(132)
        plot_tsp(plt2, x_coord, W_val, W_sol, 'Prediction: {:.3f}'.format(W_to_tour_len(W_sol, W_val)))
        plt3 = f.add_subplot(133)
        plot_tsp(plt3, x_coord, W_val, W_bs, 'Beamsearch: {:.3f}'.format(W_to_tour_len(W_bs, W_val)))
        plt.show()


# def plot_predictions_err_idx(x_nodes_coord, x_edges_values, y_edges, y_pred_edges, err_idx, num_plots=3):
#     y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
#     y = y.argmax(dim=3)  # B x V x V
#     # Plot error predictions
#     errors = err_idx.nonzero().squeeze()
#     num_plots_err = num_plots
#     if errors.size() == torch.Size([]):
#         num_plots_err = -1
#     elif num_plots > errors.size(0):
#         num_plots_err = errors.size(0)
#     print("ERROR PREDICTIONS: Plotting {} figures".format(num_plots_err))
#     for idx_plot in range(num_plots_err):
#         f = plt.figure(idx_plot, figsize=(10, 5))
#         idx = errors[idx_plot]
#         x_coord = x_nodes_coord[idx].cpu().numpy()
#         W_val = x_edges_values[idx].cpu().numpy()
#         W_target = y_edges[idx].cpu().numpy()
#         W_sol = y[idx].cpu().numpy()
#         plt1 = f.add_subplot(121)
#         plot_tsp(plt1, x_coord, W_val, W_target, 'Groundtruth: {:.3f}'.format(W_to_tour_len(W_target, W_val)))
#         plt2 = f.add_subplot(122)
#         plot_tsp(plt2, x_coord, W_val, W_sol, 'Prediction: {:.3f} (Error)'.format(W_to_tour_len(W_sol, W_val)))
#         plt.show()
#     # Plot successful predictions
#     successes = (1 - err_idx).nonzero().squeeze()
#     num_plots_suc = num_plots
#     if successes.size() == torch.Size([]):
#         num_plots_suc = -1
#     elif num_plots > successes.size(0):
#         num_plots_suc = successes.size(0)
#     print("SUCCESSFUL PREDICTIONS: Plotting {} figures".format(num_plots_suc))
#     for idx_plot in range(num_plots_suc):
#         f = plt.figure(idx_plot, figsize=(10, 5))
#         idx = successes[idx_plot]
#         x_coord = x_nodes_coord[idx].cpu().numpy()
#         W_val = x_edges_values[idx].cpu().numpy()
#         W_target = y_edges[idx].cpu().numpy()
#         W_sol = y[idx].cpu().numpy()
#         plt1 = f.add_subplot(121)
#         plot_tsp(plt1, x_coord, W_val, W_target, 'Groundtruth: {:.3f}'.format(W_to_tour_len(W_target, W_val)))
#         plt2 = f.add_subplot(122)
#         plot_tsp(plt2, x_coord, W_val, W_sol, 'Prediction: {:.3f} (Success)'.format(W_to_tour_len(W_sol, W_val)))
#         plt.show()


# def plot_predictions_err_idx_beamsearch(x_nodes_coord, x_edges_values, y_edges, y_pred_edges, bs_nodes, err_idx, num_plots=3):
#     y = F.softmax(y_pred_edges, dim=3)  # B x V x V x voc_edges
#     y = y.argmax(dim=3)  # B x V x V
#     # Plot error predictions
#     errors = err_idx.nonzero().squeeze()
#     num_plots_err = num_plots
#     if errors.size() == torch.Size([]):
#         num_plots_err = -1
#     elif num_plots > errors.size(0):
#         num_plots_err = errors.size(0)
#     print("ERROR PREDICTIONS: Plotting {} figures".format(num_plots_err))
#     for idx_plot in range(num_plots_err):
#         f = plt.figure(idx_plot, figsize=(15, 5))
#         idx = errors[idx_plot]
#         x_coord = x_nodes_coord[idx].cpu().numpy()
#         W_val = x_edges_values[idx].cpu().numpy()
#         W_target = y_edges[idx].cpu().numpy()
#         W_sol = y[idx].cpu().numpy()
#         W_bs = tour_nodes_to_W(bs_nodes[idx].cpu().numpy())
#         plt1 = f.add_subplot(131)
#         plot_tsp(plt1, x_coord, W_val, W_target, 'Groundtruth: {:.3f}'.format(W_to_tour_len(W_target, W_val)))
#         plt2 = f.add_subplot(132)
#         plot_tsp(plt2, x_coord, W_val, W_sol, 'Prediction: {:.3f} (Error)'.format(W_to_tour_len(W_sol, W_val)))
#         plt3 = f.add_subplot(133)
#         plot_tsp(plt3, x_coord, W_val, W_bs, 'Beamsearch: {:.3f}'.format(W_to_tour_len(W_bs, W_val)))
#         plt.show()
#     # Plot successful predictions
#     successes = (1 - err_idx).nonzero().squeeze()
#     num_plots_suc = num_plots
#     if successes.size() == torch.Size([]):
#         num_plots_suc = -1
#     elif num_plots > successes.size(0):
#         num_plots_suc = successes.size(0)
#     print("SUCCESSFUL PREDICTIONS: Plotting {} figures".format(num_plots_suc))
#     for idx_plot in range(num_plots_suc):
#         f = plt.figure(idx_plot, figsize=(15, 5))
#         idx = successes[idx_plot]
#         x_coord = x_nodes_coord[idx].cpu().numpy()
#         W_val = x_edges_values[idx].cpu().numpy()
#         W_target = y_edges[idx].cpu().numpy()
#         W_sol = y[idx].cpu().numpy()
#         W_bs = tour_nodes_to_W(bs_nodes[idx].cpu().numpy())
#         plt1 = f.add_subplot(131)
#         plot_tsp(plt1, x_coord, W_val, W_target, 'Groundtruth: {:.3f}'.format(W_to_tour_len(W_target, W_val)))
#         plt2 = f.add_subplot(132)
#         plot_tsp(plt2, x_coord, W_val, W_sol, 'Prediction: {:.3f} (Success)'.format(W_to_tour_len(W_sol, W_val)))
#         plt3 = f.add_subplot(133)
#         plot_tsp(plt3, x_coord, W_val, W_bs, 'Beamsearch: {:.3f}'.format(W_to_tour_len(W_bs, W_val)))
#         plt.show()


if __name__ == "__main__":
    from utils.google_tsp_reader import GoogleTSPReader

    num_nodes = 20
    batch_size = 50
    filepath = "./data/tsp5.txt"
    dataset = GoogleTSPReader(num_nodes, batch_size, filepath)
    batch = next(iter(dataset))  # Generate a batch of TSPs

    idx = 0
    f = plt.figure(figsize=(5, 5))
    a = f.add_subplot(111)
    plot_tsp(a, batch.nodes_coord[idx], batch.edges_values[idx], batch.edges_target[idx])
