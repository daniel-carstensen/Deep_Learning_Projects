import torch


class GeneralizedLogistic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l, u, g):
        """
        Computes the generalized logistic function

        Arguments
        ---------
        ctx: A PyTorch context object
        x: (Tensor) of size (T x n), the input features
        l, u, and g: (scalar tensors) representing the generalized logistic function parameters.

        Returns
        -------
        y: (Tensor) of size (T x n), the outputs of the generalized logistic operator

        """
        ctx.save_for_backward(x, l, u, g)

        exponent = torch.exp(torch.mul(-g, x))
        exponent_1 = torch.add(exponent, 1)
        frac = torch.div(u - l, exponent_1)
        y = torch.add(l, frac)

        return y

    @staticmethod
    def backward(ctx, dzdy):
        """
        back-propagate the gradients with respect to the inputs

        Arguments
        ----------
        ctx: a PyTorch context object
        dzdy (Tensor): of size (T x n), the gradients with respect to the outputs y

        Returns
        -------
        dzdx (Tensor): of size (T x n), the gradients with respect to x
        dzdl, dzdu, and dzdg: the gradients with respect to the generalized logistic parameters
        """
        x, l, u, g = ctx.saved_tensors

        exponent = torch.exp(torch.mul(-g, x))
        exponent_1 = torch.add(exponent, 1)
        exponent_sqr = torch.square(exponent_1)
        exponent_g = torch.mul(g, exponent)
        exponent_x = torch.mul(x, exponent)

        dydx = torch.div(torch.mul(exponent_g, u - l), exponent_sqr)
        dydu = torch.div(1, exponent_1)
        dydl = torch.sub(1, torch.div(1, exponent_1))
        dydg = torch.div(torch.mul(exponent_x, u - l), exponent_sqr)

        dzdx = torch.mul(dzdy, dydx)
        dzdu = torch.mul(dzdy, dydu)
        dzdl = torch.mul(dzdy, dydl)
        dzdg = torch.mul(dzdy, dydg)

        return dzdx, dzdl, dzdu, dzdg
