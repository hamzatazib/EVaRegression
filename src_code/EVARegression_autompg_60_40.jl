using CSV, Random, LinearAlgebra, DataFrames, StatsBase

# Read data
train = CSV.read("/pool001/htazi/MLProject/auto_mpg_train_1.csv")
test = CSV.read("/pool001/htazi/MLProject/auto_mpg_test_1.csv")

#From DataFrames to Matrices
#input data
X = convert(Matrix, train[:,1:7])
X_test = convert(Matrix, test[:,1:7])

#demand
Y = train[:, 8]
Y_test = test[:,8]

n, p = size(X)

function objective_l2(X,Y,s,w,alpha, lambda)
    """
    alpha: fraction of data that constitutes the training set ∈ [0,1]
    """
    n = size(X)[1]
    return lambda*dot(w, w) + s*log(sum(exp(1/s*(dot(w,X[i,:])-Y[i])^2) for i=1:n)/(n*alpha))
end

function grad_sobjective_l2(X,Y,s,w,alpha,lambda) # Not affected bY lambda but still putting it for function homogeneity issue
    n = size(X)[1]
    return log(sum(exp(1/s*(dot(w,X[i,:])-Y[i])^2) for i=1:n)/(n*alpha)) - sum(((dot(w,X[i,:])-Y[i])^2)*exp(1/s*(dot(w,X[i,:])-Y[i])^2) for i=1:n)/sum(exp(1/s*(dot(w,X[i,:])-Y[i])^2) for i=1:n)/s
end

function grad_wobjective_l2(X,Y,s,w,alpha,lambda)
    n = size(X)[1]
    return 2*lambda*w + 2*sum(((dot(w,X[i,:])-Y[i]))*exp(1/s*(dot(w,X[i,:])-Y[i])^2)*X[i,:] for i=1:n)/sum(exp(1/s*(dot(w,X[i,:])-Y[i])^2) for i=1:n)
end

function grad_descent_EVaR_l2(X, Y, s_0, w_0, alpha, lambda, c_s, c_w, epsilon)
    """
    c_s: constant learning rate associated to the grad_ w.r.t s
    c_w: constant learning rate associated to the grad_ w.r.t w
    epsilon: 
    """
    #println(objective_l2(X,Y,s_0,w_0,alpha,lambda))
    list_f = [objective_l2(X,Y,s_0,w_0,alpha,lambda)]
    s=s_0; w=w_0
    n_grad = dot(grad_wobjective_l2(X,Y,s,w,alpha,lambda), grad_wobjective_l2(X,Y,s,w,alpha,lambda)) + dot(grad_sobjective_l2(X,Y,s,w,alpha,lambda),grad_sobjective_l2(X,Y,s,w,alpha,lambda))
    k = 1
    while n_grad > epsilon
        s = s - c_s*grad_sobjective_l2(X,Y,s,w,alpha,lambda)
        s = max(0.000000001,s)
        w = w - c_w*grad_wobjective_l2(X,Y,s,w,alpha,lambda)
        n_grad = dot(grad_wobjective_l2(X,Y,s,w,alpha,lambda), grad_wobjective_l2(X,Y,s,w,alpha,lambda)) + dot(grad_sobjective_l2(X,Y,s,w,alpha,lambda),grad_sobjective_l2(X,Y,s,w,alpha,lambda))
        push!(list_f, objective_l2(X,Y,s,w,alpha,lambda))
        if k == 30000
            break
        end
        k += 1
    end
    list_f, s, w
end

function nesterov_grad_descent_EVaR_l2(X, Y, s_0, w_0, alpha, lambda, c_s, c_w, epsilon)
       """
    c_s: constant learning rate associated to the grad_ w.r.t s
    c_w: constant learning rate associated to the grad_ w.r.t w
    epsilon: stopping grad norm
    mu: momentum
    """
    #println(objective_l2(X,Y,s_0,w_0,alpha,lambda))
    list_f = [objective_l2(X,Y,s_0,w_0,alpha,lambda)]
    s=s_0; w=w_0
    n_grad = dot(grad_wobjective_l2(X,Y,s,w,alpha,lambda), grad_wobjective_l2(X,Y,s,w,alpha,lambda)) + dot(grad_sobjective_l2(X,Y,s,w,alpha,lambda),grad_sobjective_l2(X,Y,s,w,alpha,lambda))
    k=1
    while n_grad > epsilon
        mu = k/(k+3)
        if k == 30000
            break
        end
        s = (s - c_s*grad_sobjective_l2(X,Y,s,w,alpha,lambda))*(1+mu) - mu*s
        s = max(0.000000001,s)
        w = (w - c_w*grad_wobjective_l2(X,Y,s,w,alpha,lambda))*(1+mu) - mu*w
        n_grad = dot(grad_wobjective_l2(X,Y,s,w,alpha,lambda), grad_wobjective_l2(X,Y,s,w,alpha,lambda)) + dot(grad_sobjective_l2(X,Y,s,w,alpha,lambda),grad_sobjective_l2(X,Y,s,w,alpha,lambda))
        push!(list_f, objective_l2(X,Y,s,w,alpha,lambda))
        k = k+1
    end
    list_f, s, w
end

function objective_l1(X,Y,s,w,alpha, lambda)
    """
    alpha: fraction of data that constitutes the training set ∈ [0,1]
    """
    n = size(X)[1]
    return lambda*dot(w, w) + s*log(sum(exp(1/s*abs(dot(w,X[i,:])-Y[i])) for i=1:n)/(n*alpha))
end

function grad_sobjective_l1(X,Y,s,w,alpha,lambda) # Not affected bY lambda but still putting it for function homogeneity issue
    n = size(X)[1]
    return log(sum(exp(1/s*abs(dot(w,X[i,:])-Y[i])) for i=1:n)/(n*alpha)) - sum(abs(dot(w,X[i,:])-Y[i])*exp(1/s*(abs(dot(w,X[i,:])-Y[i]))) for i=1:n)/sum(exp(1/s*abs(dot(w,X[i,:])-Y[i])) for i=1:n)/s
end

function grad_wobjective_l1(X,Y,s,w,alpha,lambda)
    n = size(X)[1]
    return 2*lambda*w + sum(sign.((dot(w,X[i,:])-Y[i]))*exp(1/s*(abs(dot(w,X[i,:])-Y[i])))*X[i,:] for i=1:n)/sum(exp(1/s*(abs(dot(w,X[i,:])-Y[i]))) for i=1:n)
end

function grad_descent_EVaR_l1(X, Y, s_0, w_0, alpha, lambda, c_s, c_w, epsilon)
    """
    c_s: constant learning rate associated to the grad_ w.r.t s
    c_w: constant learning rate associated to the grad_ w.r.t w
    epsilon: 
    """
    #println(objective_l1(X,Y,s_0,w_0,alpha,lambda))
    list_f = [objective_l1(X,Y,s_0,w_0,alpha,lambda)]
    s=s_0; w=w_0
    n_grad = dot(grad_wobjective_l1(X,Y,s,w,alpha,lambda), grad_wobjective_l1(X,Y,s,w,alpha,lambda)) + dot(grad_sobjective_l1(X,Y,s,w,alpha,lambda),grad_sobjective_l1(X,Y,s,w,alpha,lambda))
    k=1
    while n_grad > epsilon
        s = s - c_s*grad_sobjective_l1(X,Y,s,w,alpha,lambda)
        s = max(0.000000001,s)
        w = w - c_w*grad_wobjective_l1(X,Y,s,w,alpha,lambda)
        n_grad = dot(grad_wobjective_l1(X,Y,s,w,alpha,lambda), grad_wobjective_l1(X,Y,s,w,alpha,lambda)) + dot(grad_sobjective_l1(X,Y,s,w,alpha,lambda),grad_sobjective_l1(X,Y,s,w,alpha,lambda))
        push!(list_f, objective_l1(X,Y,s,w,alpha,lambda))
        if k == 30000
            break
        end
        k += 1
    end
    list_f, s, w
end

function nesterov_grad_descent_EVaR_l1(X, Y, s_0, w_0, alpha, lambda, c_s, c_w, epsilon)
        """
    c_s: constant learning rate associated to the grad_ w.r.t s
    c_w: constant learning rate associated to the grad_ w.r.t w
    epsilon:
    """
    #println(objective_l1(X,Y,s_0,w_0,alpha,lambda))
    list_f = [objective_l1(X,Y,s_0,w_0,alpha,lambda)]
    s=s_0; w=w_0
    n_grad = dot(grad_wobjective_l1(X,Y,s,w,alpha,lambda), grad_wobjective_l1(X,Y,s,w,alpha,lambda)) + dot(grad_sobjective_l1(X,Y,s,w,alpha,lambda),grad_sobjective_l1(X,Y,s,w,alpha,lambda))
    k=1
    while n_grad > epsilon
        mu = k/(k+3)
        s = (s - c_s*grad_sobjective_l1(X,Y,s,w,alpha,lambda))*(1+mu) - mu*s
        s = max(0.000000001,s)
        w = (1+mu).*(w - c_w*grad_wobjective_l1(X,Y,s,w,alpha,lambda)) - mu.*w
        n_grad = dot(grad_wobjective_l1(X,Y,s,w,alpha,lambda), grad_wobjective_l1(X,Y,s,w,alpha,lambda)) + dot(grad_sobjective_l1(X,Y,s,w,alpha,lambda),grad_sobjective_l1(X,Y,s,w,alpha,lambda))
        push!(list_f, objective_l1(X,Y,s,w,alpha,lambda))
        if k == 30000
            break
        end
        k += 1
    end
    list_f, s, w
end

function fit_least_squares(X_i, Y_i, w_opt)
    return (Y_i - dot(w_opt, X_i))^2
end

function fit_least_absolute_values(X_i, Y_i, w_opt)
    return abs(Y_i - dot(w_opt, X_i))
end

function calculate_MSE(X, Y, w_opt)
    return mean((X*w_opt-Y).^2)
end

function calculate_MAE(X, Y, w_opt)
    return mean(broadcast(abs, X*w_opt-Y))
end

function get_training_and_validation_indices(X, Y, w_opt, train_is_worse, k)
    """
    train_is_worse: boolean, true if the training set contains the worst errors, false if it's the validation
    k: number of observations in the training set
    """
    n,p = size(X)
    least_squares_df = DataFrame()
    least_squares_df.Obs_Index = 1:n
    least_squares_df.LS_Value = [fit_least_squares(X[i,:], Y[i], w_opt) for i=1:n]
    least_squares_sorted = sort!(least_squares_df, :LS_Value, rev=true)
    if train_is_worse == true
        train_indices = least_squares_sorted[1:k, :Obs_Index]
        val_indices = least_squares_sorted[k+1:n, :Obs_Index]
    else
        val_indices = least_squares_sorted[1:k, :Obs_Index]
        train_indices = least_squares_sorted[k+1:n, :Obs_Index]
    end
    
    return train_indices, val_indices
end

function fit_EVaR_l2(X, Y)
    n, p = size(X)
    k = floor(Int, 0.6*n)
    alpha = k/n
    epsilon=0.001
    # Initial values, we use RLS weights as warm start
    w_0 = ones(p)./100
    s_0 = 0.9


    best_lambda_EVaR = 0
    best_c_s_EVaR = 0
    best_c_w_EVaR = 0
    best_MSE_EVaR = 1000000
    println(best_MSE_EVaR)

    for lambda_EVaR in [0, 0.01, 0.1, 0.2, 1, 10, 20, 50]
        for c_s_EVaR in [0.0001, 0.001, 0.01, 0.1]
            for c_w_EVaR in [0.0001, 0.001, 0.01, 0.1]
                list_f_opt, s_opt, w_opt_EVaR = nesterov_grad_descent_EVaR_l2(X,Y,s_0,w_0,alpha,lambda_EVaR,c_s_EVaR,c_w_EVaR,epsilon)
                train_indices_EVaR, val_indices_EVaR = get_training_and_validation_indices(
                    X, Y, w_opt_EVaR, true, k
                )
                X_train_EVaR, Y_train_EVaR = X[train_indices_EVaR, :], Y[train_indices_EVaR]
                X_val_EVaR, Y_val_EVaR = X[val_indices_EVaR, :], Y[val_indices_EVaR]
                MSE_EVaR = calculate_MSE(X_val_EVaR, Y_val_EVaR, w_opt_EVaR)
                if MSE_EVaR < best_MSE_EVaR
                    best_lambda_EVaR = lambda_EVaR
                    best_c_s_EVaR = c_s_EVaR
                    best_c_w_EVaR = c_w_EVaR
                    best_MSE_EVaR = MSE_EVaR
                    println("lambda=", lambda_EVaR, "\t c_s_EVaR=", c_s_EVaR, "\t c_w_EVaR=", c_w_EVaR, "\t MSE=", MSE_EVaR)
                end
            end
        end
    end
    println("Best lambda_EVaR = ", best_lambda_EVaR, "\t c_s_EVaR=", best_c_s_EVaR, "\t c_w_EVaR=", best_c_w_EVaR)
    return best_lambda_EVaR, best_c_s_EVaR, best_c_w_EVaR
end

# Fitting Stable regression with best lambda
n, p = size(X)
k = floor(Int, 0.6*n)
alpha = k/n
epsilon=0.001
# Initial values, we use RLS weights as warm start
w_0 = ones(p)./100
s_0 = 0.9
best_lambda_EVaR, best_c_s_EVaR, best_c_w_EVaR = fit_EVaR_l2(X, Y)
list_f_opt, s_opt, w_opt_EVaR = nesterov_grad_descent_EVaR_l2(
    X, Y, s_0, w_0, alpha, best_lambda_EVaR, best_c_s_EVaR, best_c_w_EVaR, epsilon
)
train_indices_EVaR, val_indices_EVaR = get_training_and_validation_indices(
    X, Y, w_opt_EVaR, true, k
)
X_train_EVaR, Y_train_EVaR = X[train_indices_EVaR, :], Y[train_indices_EVaR]
X_val_EVaR, Y_val_EVaR = X[val_indices_EVaR, :], Y[val_indices_EVaR]
println("Fitted EVaRegression l_2 with best possible lambda_EVaR=", best_lambda_EVaR,
    "  c_s_EVaR=", best_c_s_EVaR, "  c_w_EVaR=", best_c_w_EVaR)



function fit_EVaR_l2_warm_start(X, Y)
    n, p = size(X)
    k = floor(Int, 0.6*n)
    alpha = k/n
    epsilon=0.001
    # Initial values, we use RLS weights as warm start
    w_0 = [-0.08824811444178021, -0.02238101900183253, -0.11067152666290898, -0.44752936396063664,
        -0.014389896339926345, 0.3144806260901593, 0.12942995625812015]
    s_0 = 0.9


    best_lambda_EVaR_with_warmstart = 0
    best_c_s_EVaR_with_warmstart = 0
    best_c_w_EVaR_with_warmstart = 0
    best_MSE_EVaR_with_warmstart = 1000000
    for lambda_EVaR_with_warmstart in [0, 0.01, 0.1, 0.2, 1, 10, 20, 50]
        for c_s_EVaR_with_warmstart in [0.0001, 0.001, 0.01, 0.1]
            for c_w_EVaR_with_warmstart in [0.0001, 0.001, 0.01, 0.1]
                list_f_opt, s_opt, w_opt_EVaR_with_warmstart = nesterov_grad_descent_EVaR_l2(
                    X,Y,s_0,w_0,alpha,lambda_EVaR_with_warmstart,c_s_EVaR_with_warmstart,c_w_EVaR_with_warmstart,epsilon
                )
                train_indices_EVaR_with_warmstart, val_indices_EVaR_with_warmstart = get_training_and_validation_indices(
                    X, Y, w_opt_EVaR_with_warmstart, true, k
                )
                X_train_EVaR_with_warmstart, Y_train_EVaR_with_warmstart = X[train_indices_EVaR_with_warmstart, :], Y[train_indices_EVaR_with_warmstart]
                X_val_EVaR_with_warmstart, Y_val_EVaR_with_warmstart = X[val_indices_EVaR_with_warmstart, :], Y[val_indices_EVaR_with_warmstart]
                MSE_EVaR_with_warmstart = calculate_MSE(X_val_EVaR_with_warmstart, Y_val_EVaR_with_warmstart, w_opt_EVaR_with_warmstart)
                if MSE_EVaR_with_warmstart < best_MSE_EVaR_with_warmstart
                    best_lambda_EVaR_with_warmstart = lambda_EVaR_with_warmstart
                    best_c_s_EVaR_with_warmstart = c_s_EVaR_with_warmstart
                    best_c_w_EVaR_with_warmstart = c_w_EVaR_with_warmstart
                    best_MSE_EVaR_with_warmstart = MSE_EVaR_with_warmstart
                    println("lambda=", lambda_EVaR_with_warmstart, "\t c_s_EVaR_with_warmstart=",
                        c_s_EVaR_with_warmstart, "\t c_w_EVaR_with_warmstart=",
                        c_w_EVaR_with_warmstart, "\t MSE=", MSE_EVaR_with_warmstart)
                end
            end
        end
    end
    return best_lambda_EVaR_with_warmstart, best_c_s_EVaR_with_warmstart, best_c_w_EVaR_with_warmstart
end


n, p = size(X)
k = floor(Int, 0.6*n)
alpha = k/n
epsilon=0.001
# Initial values, we use RLS weights as warm start
w_0 = [-0.08824811444178021, -0.02238101900183253, -0.11067152666290898, -0.44752936396063664,
        -0.014389896339926345, 0.3144806260901593, 0.12942995625812015]
s_0 = 0.9
best_lambda_EVaR_with_warmstart, best_c_s_EVaR_with_warmstart, best_c_w_EVaR_with_warmstart = fit_EVaR_l2_warm_start(X, Y)
println("Best lambda_EVaR_with_warmstart = ", best_lambda_EVaR_with_warmstart,
    "\t c_s_EVaR_with_warmstart=", best_c_s_EVaR_with_warmstart,
    "\t c_w_EVaR_with_warmstart=", best_c_w_EVaR_with_warmstart)

# Fitting Stable regression with best lambda
list_f_opt, s_opt, w_opt_EVaR_with_warmstart = nesterov_grad_descent_EVaR_l2(
    X, Y, s_0, w_0, alpha, best_lambda_EVaR_with_warmstart,
    best_c_s_EVaR_with_warmstart, best_c_w_EVaR_with_warmstart, epsilon
)
train_indices_EVaR_with_warmstart, val_indices_EVaR_with_warmstart = get_training_and_validation_indices(
    X, Y, w_opt_EVaR_with_warmstart, true, k
)
X_train_EVaR_with_warmstart, Y_train_EVaR_with_warmstart = X[train_indices_EVaR_with_warmstart, :], Y[train_indices_EVaR_with_warmstart]
X_val_EVaR_with_warmstart, Y_val_EVaR_with_warmstart = X[val_indices_EVaR_with_warmstart, :], Y[val_indices_EVaR_with_warmstart]
println("Fitted EVaRegression l_2 with best possible lambda_EVaR_with_warmstart=", best_lambda_EVaR_with_warmstart,
    "  c_s_EVaR_with_warmstart=", best_c_s_EVaR_with_warmstart,
    "  c_w_EVaR_with_warmstart=", best_c_w_EVaR_with_warmstart)





function fit_EVaR_l1(X,Y)
    n, p = size(X)
    k = floor(Int, 0.6*n)
    alpha = k/n
    epsilon=0.001
    # Initial values, we use random weights as warm start
    w_0 = ones(p)./100
    s_0 = 0.9


    best_lambda_EVaR_l1 = 0
    best_c_s_EVaR_l1 = 0
    best_c_w_EVaR_l1 = 0
    best_MSE_EVaR_l1 = 1000000
    for lambda_EVaR_l1 in [0, 0.01, 0.1, 0.2, 1, 10, 20, 50]
        for c_s_EVaR_l1 in [0.0001, 0.001, 0.01, 0.1]
            for c_w_EVaR_l1 in [0.0001, 0.001, 0.01, 0.1]
                list_f_opt, s_opt, w_opt_EVaR_l1 = nesterov_grad_descent_EVaR_l1(X,Y,s_0,w_0,alpha,lambda_EVaR_l1,c_s_EVaR_l1,c_w_EVaR_l1,epsilon)
                train_indices_EVaR_l1, val_indices_EVaR_l1 = get_training_and_validation_indices(
                    X, Y, w_opt_EVaR_l1, true, k
                )
                X_train_EVaR_l1, Y_train_EVaR_l1 = X[train_indices_EVaR_l1, :], Y[train_indices_EVaR_l1]
                X_val_EVaR_l1, Y_val_EVaR_l1 = X[val_indices_EVaR_l1, :], Y[val_indices_EVaR_l1]
                MSE_EVaR_l1 = calculate_MSE(X_val_EVaR_l1, Y_val_EVaR_l1, w_opt_EVaR_l1)
                if MSE_EVaR_l1 < best_MSE_EVaR_l1
                    best_lambda_EVaR_l1 = lambda_EVaR_l1
                    best_c_s_EVaR_l1 = c_s_EVaR_l1
                    best_c_w_EVaR_l1 = c_w_EVaR_l1
                    best_MSE_EVaR_l1 = MSE_EVaR_l1
                    println("lambda=", lambda_EVaR_l1, "\t c_s_EVaR_l1=", c_s_EVaR_l1, "\t c_w_EVaR_l1=", c_w_EVaR_l1, "\t MSE=", MSE_EVaR_l1)
                else
                    println(lambda_EVaR_l1, "    ",c_s_EVaR_l1, "    ", c_w_EVaR_l1)
                end
            end
        end
    end
    println("Best lambda_EVaR_l1 = ", best_lambda_EVaR_l1, "\t c_s_EVaR_l1=", best_c_s_EVaR_l1, "\t c_w_EVaR_l1=", best_c_w_EVaR_l1)
    return best_lambda_EVaR_l1, best_c_s_EVaR_l1, best_c_w_EVaR_l1
end

# Fitting Stable regression with best lambda
n, p = size(X)
k = floor(Int, 0.6*n)
alpha = k/n
epsilon=0.001
# Initial values, we use random weights as warm start
w_0 = ones(p)./100
s_0 = 0.9
best_lambda_EVaR_l1, best_c_s_EVaR_l1, best_c_w_EVaR_l1 = fit_EVaR_l1(X,Y)
list_f_opt, s_opt, w_opt_EVaR_l1 = nesterov_grad_descent_EVaR_l1(
    X, Y, s_0, w_0, alpha, best_lambda_EVaR_l1, best_c_s_EVaR_l1, best_c_w_EVaR_l1, epsilon
)
train_indices_EVaR_l1, val_indices_EVaR_l1 = get_training_and_validation_indices(
    X, Y, w_opt_EVaR_l1, true, k
)
X_train_EVaR_l1, Y_train_EVaR_l1 = X[train_indices_EVaR_l1, :], Y[train_indices_EVaR_l1]
X_val_EVaR_l1, Y_val_EVaR_l1 = X[val_indices_EVaR_l1, :], Y[val_indices_EVaR_l1]
println("Fitted EVaR_l1egression l_1 with best possible lambda_EVaR_l1=", best_lambda_EVaR_l1,
    "  c_s_EVaR_l1=", best_c_s_EVaR_l1, "  c_w_EVaR_l1=", best_c_w_EVaR_l1)





function fit_EVaR_l1_warmstart(X, Y)
    n, p = size(X)
    k = floor(Int, 0.6*n)
    alpha = k/n
    epsilon=0.001
    # Initial values, we use RLS weights as warm start
    w_0 = [-0.08824811444178021, -0.02238101900183253, -0.11067152666290898, -0.44752936396063664,
        -0.014389896339926345, 0.3144806260901593, 0.12942995625812015]
    s_0 = 0.9


    best_lambda_EVaR_l1_with_warmstart = 0
    best_c_s_EVaR_l1_with_warmstart = 0
    best_c_w_EVaR_l1_with_warmstart = 0
    best_MSE_EVaR_l1_with_warmstart = 1000000
    for lambda_EVaR_l1_with_warmstart in [0, 0.01, 0.1, 0.2, 1, 10, 20, 50]
        for c_s_EVaR_l1_with_warmstart in [0.0001, 0.001, 0.01, 0.1]
            for c_w_EVaR_l1_with_warmstart in [0.0001, 0.001, 0.01, 0.1]
                list_f_opt, s_opt, w_opt_EVaR_l1_with_warmstart = nesterov_grad_descent_EVaR_l1(
                    X,Y,s_0,w_0,alpha,lambda_EVaR_l1_with_warmstart,c_s_EVaR_l1_with_warmstart,c_w_EVaR_l1_with_warmstart,epsilon
                )
                train_indices_EVaR_l1_with_warmstart, val_indices_EVaR_l1_with_warmstart = get_training_and_validation_indices(
                    X, Y, w_opt_EVaR_l1_with_warmstart, true, k
                )
                X_train_EVaR_l1_with_warmstart, Y_train_EVaR_l1_with_warmstart = X[train_indices_EVaR_l1_with_warmstart, :], Y[train_indices_EVaR_l1_with_warmstart]
                X_val_EVaR_l1_with_warmstart, Y_val_EVaR_l1_with_warmstart = X[val_indices_EVaR_l1_with_warmstart, :], Y[val_indices_EVaR_l1_with_warmstart]
                MSE_EVaR_l1_with_warmstart = calculate_MSE(X_val_EVaR_l1_with_warmstart, Y_val_EVaR_l1_with_warmstart, w_opt_EVaR_l1_with_warmstart)
                if MSE_EVaR_l1_with_warmstart < best_MSE_EVaR_l1_with_warmstart
                    best_lambda_EVaR_l1_with_warmstart = lambda_EVaR_l1_with_warmstart
                    best_c_s_EVaR_l1_with_warmstart = c_s_EVaR_l1_with_warmstart
                    best_c_w_EVaR_l1_with_warmstart = c_w_EVaR_l1_with_warmstart
                    best_MSE_EVaR_l1_with_warmstart = MSE_EVaR_l1_with_warmstart
                    println("lambda=", lambda_EVaR_l1_with_warmstart, "\t c_s_EVaR_l1_with_warmstart=",
                        c_s_EVaR_l1_with_warmstart, "\t c_w_EVaR_l1_with_warmstart=",
                        c_w_EVaR_l1_with_warmstart, "\t MSE=", MSE_EVaR_l1_with_warmstart)
                else
                    println(lambda_EVaR_l1_with_warmstart, "    ",
                        c_s_EVaR_l1_with_warmstart, "    ",
                        c_w_EVaR_l1_with_warmstart)
                end
            end
        end
    end
    return best_lambda_EVaR_l1_with_warmstart, best_c_s_EVaR_l1_with_warmstart, best_c_w_EVaR_l1_with_warmstart
end
n, p = size(X)
k = floor(Int, 0.6*n)
alpha = k/n
epsilon=0.001
# Initial values, we use RLS weights as warm start
w_0 = [-0.08824811444178021, -0.02238101900183253, -0.11067152666290898, -0.44752936396063664,
        -0.014389896339926345, 0.3144806260901593, 0.12942995625812015]
s_0 = 0.9
best_lambda_EVaR_l1_with_warmstart, best_c_s_EVaR_l1_with_warmstart, best_c_w_EVaR_l1_with_warmstart = fit_EVaR_l1_warmstart(X, Y)
println("Best lambda_EVaR_l1_with_warmstart = ", best_lambda_EVaR_l1_with_warmstart, 
    "\t c_s_EVaR_l1_with_warmstart=", best_c_s_EVaR_l1_with_warmstart, 
    "\t c_w_EVaR_l1_with_warmstart=", best_c_w_EVaR_l1_with_warmstart)

# Fitting Stable regression with best lambda
list_f_opt, s_opt, w_opt_EVaR_l1_with_warmstart = nesterov_grad_descent_EVaR_l1(
    X, Y, s_0, w_0, alpha, best_lambda_EVaR_l1_with_warmstart, 
    best_c_s_EVaR_l1_with_warmstart, best_c_w_EVaR_l1_with_warmstart, epsilon
)
train_indices_EVaR_l1_with_warmstart, val_indices_EVaR_l1_with_warmstart = get_training_and_validation_indices(
    X, Y, w_opt_EVaR_l1_with_warmstart, true, k
)
X_train_EVaR_l1_with_warmstart, Y_train_EVaR_l1_with_warmstart = X[train_indices_EVaR_l1_with_warmstart, :], Y[train_indices_EVaR_l1_with_warmstart]
X_val_EVaR_l1_with_warmstart, Y_val_EVaR_l1_with_warmstart = X[val_indices_EVaR_l1_with_warmstart, :], Y[val_indices_EVaR_l1_with_warmstart]
println("Fitted EVaR_l1egression l_1 with best possible lambda_EVaR_l1_with_warmstart=", best_lambda_EVaR_l1_with_warmstart, 
    "  c_s_EVaR_l1_with_warmstart=", best_c_s_EVaR_l1_with_warmstart, 
    "  c_w_EVaR_l1_with_warmstart=", best_c_w_EVaR_l1_with_warmstart)

println("####################################################################################################")
println("####################################################################################################")
println("####################################################################################################")
println("####################################################################################################")
println("####################################################################################################")
println("####################################################################################################")

println("########## The training scores are: ##########")
#println("The MSE for the Regularized Least Squares is : ",
#    calculate_MSE(X_train_RLS, Y_train_RLS, w_opt_RLS))
#println("The MSE for the Stable Regression is : ",
#    calculate_MSE(X_train_SR, Y_train_SR, w_opt_SR))
println("The MSE for the EVaR Regression is for epsilon = 0.001 : ", 
    calculate_MSE(X_train_EVaR, Y_train_EVaR, w_opt_EVaR))
println("The MSE for the EVaR Regression with RLS warm start is for epsilon = 0.001 : ", 
    calculate_MSE(X_train_EVaR_with_warmstart, Y_train_EVaR_with_warmstart, w_opt_EVaR_with_warmstart))
println("The MSE for the EVaR_l1 Regression is for epsilon = 0.001 : ", 
    calculate_MSE(X_train_EVaR_l1, Y_train_EVaR_l1, w_opt_EVaR_l1))
println("The MSE for the EVaR_l1 Regression with RLS warm start is for epsilon = 0.001 : ", 
    calculate_MSE(X_train_EVaR_l1_with_warmstart, Y_train_EVaR_l1_with_warmstart, w_opt_EVaR_l1_with_warmstart))

println("########## The validation scores are: ##########")
#println("The MSE for the Regularized Least Squares is : ",
#    calculate_MSE(X_val_RLS, Y_val_RLS, w_opt_RLS))
#println("The MSE for the Stable Regression is : ",
#    calculate_MSE(X_val_SR, Y_val_SR, w_opt_SR))
println("The MSE for the EVaR Regression is for epsilon = 0.001 : ", 
    calculate_MSE(X_val_EVaR, Y_val_EVaR, w_opt_EVaR))
println("The MSE for the EVaR Regression with RLS warm start is for epsilon = 0.001 : ", 
    calculate_MSE(X_val_EVaR_with_warmstart, Y_val_EVaR_with_warmstart, w_opt_EVaR_with_warmstart))
println("The MSE for the EVaR_l1 Regression is for epsilon = 0.001 : ", 
    calculate_MSE(X_val_EVaR_l1, Y_val_EVaR_l1, w_opt_EVaR_l1))
println("The MSE for the EVaR_l1 Regression with RLS warm start is for epsilon = 0.001 : ", 
    calculate_MSE(X_val_EVaR_l1_with_warmstart, Y_val_EVaR_l1_with_warmstart, w_opt_EVaR_l1_with_warmstart))

println("########## The test scores are: ##########")
#println("The MSE for the Regularized Least Squares is : ",
#    calculate_MSE(X_test, Y_test, w_opt_RLS))
#println("The MSE for the Stable Regression is : ",
#    calculate_MSE(X_test, Y_test, w_opt_SR))
println("The MSE for the EVaR Regression is for epsilon = 0.001 : ", 
    calculate_MSE(X_test, Y_test, w_opt_EVaR))
println("The MSE for the EVaR Regression with RLS warm start is for epsilon = 0.001 : ", 
    calculate_MSE(X_test, Y_test, w_opt_EVaR_with_warmstart))
println("The MSE for the EVaR_l1 Regression is for epsilon = 0.001 : ", 
    calculate_MSE(X_test, Y_test, w_opt_EVaR_l1))
println("The MSE for the EVaR_l1 Regression with RLS warm start is for epsilon = 0.001 : ", 
    calculate_MSE(X_test, Y_test, w_opt_EVaR_l1_with_warmstart))
println("####################################################################################################")
println("####################################################################################################")
println("####################################################################################################")
println("########## The training scores are: ##########")
#println("The MAE for the Regularized Least Squares is : ",
#    calculate_MAE(X_train_RLS, Y_train_RLS, w_opt_RLS))
#println("The MAE for the Stable Regression is : ",
#    calculate_MAE(X_train_SR, Y_train_SR, w_opt_SR))
println("The MAE for the EVaR Regression is for epsilon = 0.001 : ", 
    calculate_MAE(X_train_EVaR, Y_train_EVaR, w_opt_EVaR))
println("The MAE for the EVaR Regression with RLS warm start is for epsilon = 0.001 : ", 
    calculate_MAE(X_train_EVaR_with_warmstart, Y_train_EVaR_with_warmstart, w_opt_EVaR_with_warmstart))
println("The MAE for the EVaR_l1 Regression is for epsilon = 0.001 : ", 
    calculate_MAE(X_train_EVaR_l1, Y_train_EVaR_l1, w_opt_EVaR_l1))
println("The MAE for the EVaR_l1 Regression with RLS warm start is for epsilon = 0.001 : ", 
    calculate_MAE(X_train_EVaR_l1_with_warmstart, Y_train_EVaR_l1_with_warmstart, w_opt_EVaR_l1_with_warmstart))

println("########## The validation scores are: ##########")
#println("The MAE for the Regularized Least Squares is : ",
#    calculate_MAE(X_val_RLS, Y_val_RLS, w_opt_RLS))
#println("The MAE for the Stable Regression is : ",
#    calculate_MAE(X_val_SR, Y_val_SR, w_opt_SR))
println("The MAE for the EVaR Regression is for epsilon = 0.001 : ", 
    calculate_MAE(X_val_EVaR, Y_val_EVaR, w_opt_EVaR))
println("The MAE for the EVaR Regression with RLS warm start is for epsilon = 0.001 : ", 
    calculate_MAE(X_val_EVaR_with_warmstart, Y_val_EVaR_with_warmstart, w_opt_EVaR_with_warmstart))
println("The MAE for the EVaR_l1 Regression is for epsilon = 0.001 : ", 
    calculate_MAE(X_val_EVaR_l1, Y_val_EVaR_l1, w_opt_EVaR_l1))
println("The MAE for the EVaR_l1 Regression with RLS warm start is for epsilon = 0.001 : ", 
    calculate_MAE(X_val_EVaR_l1_with_warmstart, Y_val_EVaR_l1_with_warmstart, w_opt_EVaR_l1_with_warmstart))

println("########## The test scores are (MAE): ##########")
#println("The MAE for the Regularized Least Squares is : ",
#    calculate_MAE(X_test, Y_test, w_opt_RLS))
#println("The MAE for the Stable Regression is : ",
#    calculate_MAE(X_test, Y_test, w_opt_SR))
println("The MAE for the EVaR Regression is for epsilon = 0.001 : ", 
    calculate_MAE(X_test, Y_test, w_opt_EVaR))
println("The MAE for the EVaR Regression with RLS warm start is for epsilon = 0.001 : ", 
    calculate_MAE(X_test, Y_test, w_opt_EVaR_with_warmstart))
println("The MAE for the EVaR_l1 Regression is for epsilon = 0.001 : ", 
    calculate_MAE(X_test, Y_test, w_opt_EVaR_l1))
println("The MAE for the EVaR_l1 Regression with RLS warm start is for epsilon = 0.001 : ", 
    calculate_MAE(X_test, Y_test, w_opt_EVaR_l1_with_warmstart))