from mean_squared_error import MeanSquaredError
import torch


def mean_squared_error_test():
    """
     Unit tests for the MeanSquaredError autograd Function.

    PROVIDED CONSTANTS
    ------------------
    TOL (float): the absolute error tolerance for the backward mode. If any error is equal to or
                greater than TOL, is_correct is false
    DELTA (float): The difference parameter for the finite difference computation
    X1 (Tensor): size (48 x 2) denoting 72 example inputs each with 2 features
    X2 (Tensor): size (48 x 2) denoting the targets

    Returns
    -------
    is_correct (boolean): True if and only if MeanSquaredError passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx1 (float): the  error between the analytical and numerical gradients w.r.t X1
                    2. dzdx2 (float): The error between the analytical and numerical gradients w.r.t X2
    Note
    -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%
    dataset = torch.load("mean_squared_error_test.pt")
    X1 = dataset["X1"]
    X2 = dataset["X2"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    mean_squared_error = MeanSquaredError.apply
    # %%% DO NOT EDIT ABOVE %%%
    y = mean_squared_error(X1, X2)
    z = torch.mean(y)
    dzdy = torch.autograd.grad(outputs=z, inputs=y, grad_outputs=torch.ones_like(z))
    z.backward(dzdy)

    with torch.no_grad():
        dzdx1_n = torch.zeros(X1.shape)
        dzdx2_n = torch.zeros(X2.shape)

        for t in range(X1.shape[0]):
            for i in range(X1.shape[1]):
                X1_plus = X1.clone()
                X1_minus = X1.clone()

                X1_plus[t, i] = torch.add(X1_plus[t, i], DELTA)
                X1_minus[t, i] = torch.sub(X1_minus[t, i], DELTA)

                step = torch.sub(mean_squared_error(X1_plus, X2), mean_squared_error(X1_minus, X2))
                step = torch.div(step, 2 * DELTA)

                dzdx1_n[t, i] = torch.sum(torch.mul(dzdy[0], step))

        for t in range(X2.shape[0]):
            for i in range(X2.shape[1]):
                X2_plus = X2.clone()
                X2_minus = X2.clone()

                X2_plus[t, i] = torch.add(X2_plus[t, i], DELTA)
                X2_minus[t, i] = torch.sub(X2_minus[t, i], DELTA)

                step = torch.sub(mean_squared_error(X1, X2_plus), mean_squared_error(X1, X2_minus))
                step = torch.div(step, 2 * DELTA)

                dzdx2_n[t, i] = torch.sum(torch.mul(dzdy[0], step))

    err = dict()

    err_x1 = torch.max(torch.sub(X1.grad, dzdx1_n))
    err['dzdx1'] = err_x1

    err_x2 = torch.max(torch.sub(X2.grad, dzdx2_n))
    err['dzdx2'] = err_x2

    if max(err_x1, err_x2) >= TOL:
        is_correct_num = False
    else:
        is_correct_num = True

    is_correct_auto = torch.autograd.gradcheck(func=mean_squared_error, inputs=(X1, X2), eps=DELTA, atol=TOL)
    is_correct = is_correct_num and is_correct_auto

    torch.save([is_correct, err], 'mean_squared_error_test_result.pt')

    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = mean_squared_error_test()
    assert tests_passed
    print(errors)
