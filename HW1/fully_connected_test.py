import pdb

from fully_connected import FullyConnected
import torch

def fully_connected_test():
    """
    Provides Unit tests for the FullyConnected autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL (float): The error tolerance for the backward mode. If the error >= TOL, then is_correct is false
    DELTA (float): The difference parameter for the finite difference computations
    X (Tensor): of size (48 x 2), the inputs
    W (Tensor): of size (2 x 72), the weights
    B (Tensor): of size (72), the biases

    Returns
    -------
    is_correct (boolean): True if and only iff FullyConnected passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx: the error between the analytical and numerical gradients w.r.t X
                    2. dzdw (float): ... w.r.t W
                    3. dzdb (float): ... w.r.t B

    Note
    ----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%%
    dataset = torch.load("fully_connected_test.pt")
    X = dataset["X"]
    W = dataset["W"]
    B = dataset["B"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    full_connected = FullyConnected.apply
    # %%% DO NOT EDIT ABOVE
    y = full_connected(X, W, B)
    z = torch.mean(y)
    dzdy = torch.autograd.grad(outputs=z, inputs=y)
    z.backward()

    with torch.no_grad():
        dzdx_n = torch.zeros(X.shape)
        dzdb_n = torch.zeros(B.shape)
        dzdw_n = torch.zeros(W.shape)

        for t in range(X.shape[0]):
            for i in range(X.shape[1]):
                X_plus = X.clone()
                X_minus = X.clone()

                X_plus[t, i] = torch.add(X_plus[t, i], DELTA)
                X_minus[t, i] = torch.sub(X_minus[t, i], DELTA)

                step = torch.sub(full_connected(X_plus, W, B), full_connected(X_minus, W, B))
                step = torch.div(step, 2 * DELTA)

                dzdx_n[t, i] = torch.sum(torch.mul(dzdy[0], step))

        for t in range(B.shape[0]):
            B_plus = B.clone()
            B_minus = B.clone()

            B_plus[t] = torch.add(B_plus[t], DELTA)
            B_minus[t] = torch.sub(B_minus[t], DELTA)

            step = torch.sub(full_connected(X, W, B_plus), full_connected(X, W, B_minus))
            step = torch.div(step, 2 * DELTA)

            dzdb_n[t] = torch.sum(torch.mul(dzdy[0], step))

        for t in range(W.shape[0]):
            for i in range(W.shape[1]):
                W_plus = W.clone()
                W_minus = W.clone()

                W_plus[t, i] = torch.add(W_plus[t, i], DELTA)
                W_minus[t, i] = torch.sub(W_minus[t, i], DELTA)

                step = torch.sub(full_connected(X, W_plus, B), full_connected(X, W_minus, B))
                step = torch.div(step, 2 * DELTA)

                dzdw_n[t, i] = torch.sum(torch.mul(dzdy[0], step))

    err = dict()

    err_x = torch.max(torch.sub(X.grad, dzdx_n))
    err['dzdx'] = err_x

    err_b = torch.max(torch.sub(B.grad, dzdb_n))
    err['dzdb'] = err_b

    err_w = torch.max(torch.sub(W.grad, dzdw_n))
    err['dzdw'] = err_w

    is_correct_num = (max(err_x, err_b, err_w) < TOL)
    is_correct_auto = torch.autograd.gradcheck(func=full_connected, inputs=(X, W, B), eps=DELTA, atol=TOL)
    is_correct = is_correct_num and is_correct_auto

    torch.save([is_correct, err], 'fully_connected_test_results.pt' )

    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = fully_connected_test()
    assert tests_passed
    print(errors)
