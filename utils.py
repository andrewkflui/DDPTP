import numpy as np
def get_ade(predicted, ground_truth):
    """
    calculate the ADE on predicted trajectories
    :param predicted: the predicted trajectories, in form of
           [num_trajectories, predicted_length, 2]
    :param ground_truth: ground truth label of predicted trajectories,
           in form of [num_trajectories, predicted_length, 2]
    :return: average ADE
    """

    mean_ADE = np.mean(np.sqrt(np.sum((predicted-ground_truth)**2, axis=-1)))
    return mean_ADE
def get_fde(predicted, ground_truth):
    """
    calculate the FDE on predicted trajectories
    :param predicted: predicted trajectories, in form of
          [num_trajectories, predicted_length, 2]
    :param ground_truth: ground truth label of predicted trajectories,
          in form of [num_trajectories, predicted_length, 2]
    :return: average FDE
    """

    mean_FDE = np.mean(np.sqrt(
            np.sum((predicted[:, -1, :]-ground_truth[:, -1, :])**2, axis=-1)))

    return mean_FDE
def image_to_world(p):
    # x and y are swapped for the NYGC dataset's Homog matrix
    # from https://github.com/crowdbotp/OpenTraj/tree/master/datasets/GC
    # https://github.com/crowdbotp/OpenTraj/blob/master/opentraj/toolkit/loaders/loader_gcs.py
    p[:,[0,1]] = p[:,[1,0]]
    Homog = [[4.97412897e-02, -4.24730883e-02, 7.25543911e+01],
             [1.45017874e-01, -3.35678711e-03, 7.97920970e+00],
             [1.36068797e-03, -4.98339188e-05, 1.00000000e+00]]
    pp = np.stack((p[:, 0], p[:, 1], np.ones(len(p))), axis=1)
    PP = np.matmul(Homog, pp.T).T
    P_normal = PP / np.repeat(PP[:, 2].reshape((-1, 1)), 3, axis=1)
    return P_normal[:, :2]
