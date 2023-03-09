from generalized_logistic import GeneralizedLogistic
import torch


def generalized_logistic_test():
    """
    Provides Unit tests for the GeneralizedLogistic autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL1 (float): the  error tolerance for the forward mode. If the error >= TOL1, is_correct is false
    TOL2 (float): The error tolerance for the backward mode
    DELTA (float): The difference parameter for the finite differences computation
    X (Tensor): size (48 x 2) of inputs
    L, U, and G (floats): The parameter values necessary to compute the hyperbolic tangent (tanH) using
                        GeneralizedLogistic
    Returns:
    -------
    is_correct (boolean): True if and only if GeneralizedLogistic passes all unit tests
    err (Dictionary): with the following keys
                        1. y (float): The error between the forward direction and the results of pytorch's tanH
                        2. dzdx (float): the error between the analytical and numerical gradients w.r.t X
                        3. dzdl (float): ... w.r.t L
                        4. dzdu (float): ... w.r.t U
                        5. dzdg (float): .. w.r.t G
     Note
     -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%%% DO NOT EDIT BELOW %%%
    dataset = torch.load("generalized_logistic_test.pt")
    X = dataset["X"]
    L = dataset["L"]
    U = dataset["U"]
    G = dataset["G"]
    TOL1 = dataset["TOL1"]
    TOL2 = dataset["TOL2"]
    DELTA = dataset["DELTA"]
    generalized_logistic = GeneralizedLogistic.apply
    # %%%  DO NOT EDIT ABOVE %%%
    y = generalized_logistic(X, L, U, G)
    y_tanh = torch.tanh(X)
    z = torch.mean(y)
    dzdy = torch.autograd.grad(outputs=z, inputs=y)
    z.backward()

    with torch.no_grad():
        dzdx_n = torch.zeros(X.shape)

        for t in range(X.shape[0]):
            for i in range(X.shape[1]):
                X_plus = X.clone()
                X_minus = X.clone()

                X_plus[t, i] = torch.add(X_plus[t, i], DELTA)
                X_minus[t, i] = torch.sub(X_minus[t, i], DELTA)

                step = torch.sub(generalized_logistic(X_plus, L, U, G), generalized_logistic(X_minus, L, U, G))
                step = torch.div(step, 2 * DELTA)

                dzdx_n[t, i] = torch.sum(torch.mul(dzdy[0], step))

        L_plus = L.clone()
        L_minus = L.clone()

        L_plus = torch.add(L_plus, DELTA)
        L_minus = torch.sub(L_minus, DELTA)

        step = torch.sub(generalized_logistic(X, L_plus, U, G), generalized_logistic(X, L_minus, U, G))
        step = torch.div(step, 2 * DELTA)

        dzdl_n = torch.sum(torch.mul(dzdy[0], step))

        U_plus = U.clone()
        U_minus = U.clone()

        U_plus = torch.add(U_plus, DELTA)
        U_minus = torch.sub(U_minus, DELTA)

        step = torch.sub(generalized_logistic(X, L, U_plus, G), generalized_logistic(X, L, U_minus, G))
        step = torch.div(step, 2 * DELTA)

        dzdu_n = torch.sum(torch.mul(dzdy[0], step))

        G_plus = G.clone()
        G_minus = G.clone()

        G_plus = torch.add(G_plus, DELTA)
        G_minus = torch.sub(G_minus, DELTA)

        step = torch.sub(generalized_logistic(X, L, U, G_plus), generalized_logistic(X, L, U, G_minus))
        step = torch.div(step, 2 * DELTA)

        dzdg_n = torch.sum(torch.mul(dzdy[0], step))

    err = dict()

    err_y = torch.max(torch.sub(y, y_tanh))
    err['y'] = err_y

    err_x = torch.max(torch.sub(X.grad, dzdx_n))
    err['dzdx'] = err_x

    err_l = torch.max(torch.sub(L.grad, dzdl_n))
    err['dzdl'] = err_l

    err_u = torch.max(torch.sub(U.grad, dzdu_n))
    err['dzdu'] = err_u

    err_g = torch.max(torch.sub(G.grad, dzdg_n))
    err['dzdg'] = err_g

    is_correct_forward = (err_y < TOL1)
    is_correct_num = (max(err_x, err_l, err_u, err_g) < TOL2)
    is_correct_auto = torch.autograd.gradcheck(func=generalized_logistic, inputs=(X, L, U, G), eps=DELTA, atol=TOL2)

    is_correct = is_correct_forward and is_correct_num and is_correct_auto

    torch.save([is_correct, err], 'generalized_logistic_test_results.pt')

    return is_correct, err


if __name__ == '__main__':
    test_passed, errors = generalized_logistic_test()
    assert test_passed
    print(errors)
