import torch.nn.functional as F

def loss_factory(loss_type, residual_list):
    """
    Maps a loss type string to the corresponding PyTorch loss function.

    Args:
        loss_type: str
            Name of the loss function.
        residual_list: list[str]
            list of name of residuals
    return: callable
        PyTorch loss function.
    """
    # for residual in residual_list: make a single callable function as a sum of the other residuals
    if loss_type == "mse":
        return F.mse_loss
    if loss_type == "mae":
        return F.l1_loss
    if loss_type == "huber":
        return F.huber_loss
    raise NotImplementedError("Unsupported loss function")

def volume_loss(pred, target):
    # FIXME: implement volume loss/residual: the difference between the "rest" and "non-rest" volume (scalars)
    #       has to be minimized
    return

def curvature_loss(pred, target):
    # FIXME: implement extreme curvature penalization, where as a measure of curvature we use the angle between a couple
    #       of adjacent tetrahedras, for all couple of adjacent tetrahedras. (that angle can be computed as the angle
    #       between the vector normal to the surface of the tetrahedron?)
    #       We take the maximum angle and we penalize it gradually, i.e. the bigger it is the most it is penalized, and
    #       maybe set an arbitrary threshold below which the penalization is not activated (30 degrees).
    return
