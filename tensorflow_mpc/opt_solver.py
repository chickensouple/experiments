import numpy as np
import tensorflow as tf
import scipy.optimize

class DiscreteSystemModel(object):
    def __init__(self, state_dim, control_dim):
        self.state_dim = state_dim
        self.control_dim = control_dim

    def step(self, x, u):
        """
        Computes the next state given current state and action.
        This function should be implemented using differentiable tensorflow functions
        so that tf.GradientTape() can compute jacobians automatically.
        This can compute the step function for multiple states and control at the same time.
        (N is the number of states and controls to compute in parallel)
        Both x and u should have dtype=tf.float32

        Arguments:
            x {np.array} -- (N, state_dim) size array of state.
            u {np.array} -- (N, control_dim) size array of control,
        
        Returns:
            {np.array} -- (N, state_dim) size array for next_state
        """
        raise Exception("Not implemented.")

    def _check_input(self, x, u):
        assert(len(x.shape) == 2)
        assert(x.shape[1] == self.state_dim)
        assert(len(u.shape) == 2)
        assert(u.shape[1] == self.control_dim)

        assert(u.shape[0] == x.shape[0])

class ScipyMPCOptProblem(object):
    """
    This class represents an MPC control problem that is solved with scipy.optimize.minimize().
    The problem represented is

    min_{X, U} c(X, U)
    s.t.       g_i(X, U) <= 0 \forall i
               h_i(X, U) = 0 \forall i
               x_{k+1} = f(x_k, u_k) for k=0, ..., T-1

    where X and U are big vectors that are all the states and controls for all time concatenated.
    c is a generic cost function.
    g_i are generic constraint functions.
    the last constraint is the dynamics constraints and is automatically put into the problem
    when this class is inherited from.


    This is a base class that should be subclassed. 
    4 functions will need to be overriden: initial_guess(), eval_obj(), eval_ineq_constraints(), eval_eq_constraints(),
    eval_obj() and eval_ineq_constraints() and eval_eq_constraints() should be written using differentiable tensorflow functions 
    so that tf.GradientTape() can automatically compute jacobians.

    If any of the constraints are not overriden, they will default to 0 (no constraints).

    overloaded functions should not use class variables. Pass in extra variables by setting the
    self.problem_kwargs. Passed in variables must be tf.Tensors


    After this is subclassed, you can instantiate the subclass and call solve.
    Ex:
    class MyProblem(ScipyMPCOptProblem):
        def __init__(self):
            ...

        def initial_guess(self):
            ...
        
        def eval_obj(self, state, control):
            ...

        def eval_ineq_constraints(self, state, control):
            ...

    problem = MyProblem()
    sol = problem.solve()
    """
    def __init__(self, T, sys_model, model_eps=0.01):
        """
        Creates the optimization problem.
        When subclassing this class, you must call super().__init__() first thing 
        in the subclass ctor

        Arguments:
            T {int} -- the time horizon
            sys_model {DiscreteSystemModel} -- a system model used for dynamics constraints.
        
        Keyword Arguments:
            model_eps {float} -- error that the model can tolerate (default: {0.01})
        """
        self.state_dim = sys_model.state_dim
        self.control_dim = sys_model.control_dim
        self.T = T
        self.sys_model = sys_model
        self.model_eps = model_eps

        self.problem_kwargs = dict()

        # compiling jacobian calculations as static tensorflow graphs to fast computation later
        eval_model_eq_constraints = lambda state, control: self._eval_model_eq_constraints(state, control)
        self.tf_eval_model_eq_constraints = tf.function(eval_model_eq_constraints)

        eval_model_eq_constraints_jac = lambda state, control: self._compute_jac(self._eval_model_eq_constraints, state, control)
        self.tf_eval_model_eq_constraints_jac = tf.function(eval_model_eq_constraints_jac)

        eval_ineq_constraints = lambda state, control, **kwargs: self.eval_ineq_constraints(state, control, **kwargs)
        self.tf_eval_ineq_constraints = tf.function(eval_ineq_constraints)

        eval_ineq_constraints_jac = lambda state, control, **kwargs: self._compute_jac(self.eval_ineq_constraints, state, control, **kwargs)
        self.tf_eval_ineq_constraints_jac = tf.function(eval_ineq_constraints_jac)

        eval_eq_constraints = lambda state, control, **kwargs: self.eval_eq_constraints(state, control, **kwargs)
        self.tf_eval_eq_constraints = tf.function(eval_eq_constraints)

        eval_eq_constraints_jac = lambda state, control, **kwargs: self._compute_jac(self.eval_eq_constraints, state, control, **kwargs)
        self.tf_eval_eq_constraints_jac = tf.function(eval_eq_constraints_jac)

        eval_obj = lambda state, control, **kwargs: self.eval_obj(state, control, **kwargs)
        self.tf_eval_obj = tf.function(eval_obj)

        eval_obj_jac = lambda state, control, **kwargs: self._compute_jac(self.eval_obj, state, control, **kwargs)
        self.tf_eval_obj_jac = tf.function(eval_obj_jac)

    def initial_guess(self):
        """
        Return initial guess for the problem.
        Needs to return (state, control).
        [state] is a (T, state_dim) tf.Tensor (with dtype=tf.float32) 
        where the i'th row is the state for the i'th time step. 
        [control] is a (T-1, control_dim) tf.Tensor (with dtype=tf.float32) 
        where the i'th row is the control action for the i'th time step.

        [control] only has T-1 rows since, the (T-1)'th control and state
        will give the T'th state, and no more control actions will be taken after that.
        """
        raise Exception("Not implemented")

    def eval_obj(self, state, control, **kwargs):
        """
        Evalautes the objective function.
        This function should be written in tensorflow so that the gradient
        can be automatically computed.
        
        Arguments:
            state tf.Tensor -- see initial_guess() for description
            control tf.Tensor -- see initial_guess() for description
        
        Returns:
            {tf.Tensor}: scalar value of the objective function
        """
        obj = np.array([0], dtype=np.float32)
        obj = tf.convert_to_tensor(obj)
        return obj

    def eval_ineq_constraints(self, state, control, **kwargs):
        """
        Evalautes the inequality function.
        Evalutes h_i(X, U) (for the equality h_i(X, U) = 0) for all i.
        This should return a column vector where the i'th element is h_i(X, U).

        This function should be written in tensorflow so that the gradient
        can be automatically computed.
        
        Arguments:
            state tf.Tensor -- see initial_guess() for description
            control tf.Tensor -- see initial_guess() for description
        
        Returns:
            {tf.Tensor}: (N by 1) vector of constraints
        """
        constraints = tf.zeros((1, 1))
        return constraints

    def eval_eq_constraints(self, state, control, **kwargs):
        constraints = tf.zeros((1, 1))
        return constraints


    def solve(self, maxiter=21, ftol=1e-6):
        assert(self.T > 1)

        state0, control0 = self.initial_guess()
        self._check_state_and_control(state0, control0)
        x0 = self._flatten_var(state0, control0)

        constraints = [
            {"type": "ineq",
             "fun": self._scipy_ineq_constraints,
             "jac": self._scipy_ineq_constraints_jac},
            {"type": "eq",
             "fun": self._scipy_eq_constraints,
             "jac": self._scipy_eq_constraints_jac},
            # {"type": "ineq",
            #  "fun": self._scipy_model_ineq_constraints,
            #  "jac": self._scipy_model_ineq_constraints_jac}
            {"type": "eq",
             "fun": self._scipy_model_eq_constraints,
             "jac": self._scipy_model_eq_constraints_jac},
        ]

        sol = scipy.optimize.minimize(
            fun=self._scipy_obj,
            jac=self._scipy_obj_jac,
            x0=x0,
            method="SLSQP",
            constraints=constraints,
            tol=0.2,
            options={
                "ftol": ftol,
                "maxiter": maxiter,
            }
        )
        state, control = self._unflatten_var(sol.x)
        return sol, state, control

    # internal helper functions
    def _check_state_and_control(self, state, control):
        assert(len(state.shape) == 2)
        assert(state.shape[0] == self.T)
        assert(state.shape[1] == self.state_dim)

        assert(len(control.shape) == 2)
        assert(control.shape[0] == self.T-1)
        assert(control.shape[1] == self.control_dim)

    def _flatten_var(self, state, control):
        """
        Turns (state, control) [see initial_guess() for description]
        into a big N by 1 vector
        """
        state_flattened = np.reshape(state, (-1, 1))
        control_flattened = np.reshape(control, (-1, 1))
        var = np.concatenate([state_flattened, control_flattened], axis=0)
        return var

    def _unflatten_var(self, var):
        """
        Turns a big N by 1 vector into (state, control) [see initial_guess() for description]
        """
        assert((len(var) + self.control_dim) % (self.control_dim + self.state_dim) == 0)
        T = int((len(var) + self.control_dim) / (self.control_dim + self.state_dim))
        state_flattened = var[:(self.state_dim*T)]
        control_flattened = var[(self.state_dim*T):]
        state = np.reshape(state_flattened, (T, self.state_dim))
        control = np.reshape(control_flattened, (T-1, self.control_dim))
        return state, control

    def _convert_to_tensor(self, state, control):
        state = tf.cast(tf.convert_to_tensor(state), tf.float32)
        control = tf.cast(tf.convert_to_tensor(control), tf.float32)
        return state, control

    def _convert_to_array(self, tensor):
        return np.array(tensor, dtype=np.float64)

    def _eval_model_ineq_constraints(self, state, control):
        """
        Returns the dynamics model consistency constraints.
        |x_{k+1} - f(x_k, u_k)| < eps for k=0, ..., T-1
        """

        pred_state = self.sys_model.step(state[:-1, :], control)
        constraints_lb = state[1:, :] - pred_state - self.model_eps
        constraints_ub = pred_state - self.model_eps - state[1:, :]
        constraints = tf.reshape(tf.stack([constraints_lb, constraints_ub]),(-1, 1))
        return constraints

    def _eval_model_eq_constraints(self, state, control):
        """
        returns the dynamics model consistency constraints
        x_{k+1} = f(x_k, u_k) for k=0, ..., T-1        
        """
        pred_state = self.sys_model.step(state[:-1, :], control)
        constraints = pred_state - state[1:, :]
        constraints = tf.reshape(constraints, (-1, 1))
        return constraints

    def _compute_jac(self, func, state, control, **kwargs):
        """
        Computes the jacobian of a function (func)
        at the location (state, control).
        func must have a function signature that looks like
        def func(state, control).
        It also must be written using tensorflow operations so that
        tf.GradientTape can compute the derivatives.
        """
        # variables for tf.GradientTape() to keep track of
        # and to take the jacobian with respect to
        variables = [state, control]

        with tf.GradientTape() as tape:
            tape.watch(variables)
            output = func(state, control, **kwargs)

        jac = tape.jacobian(
            target=output, 
            sources=variables, 
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return jac


    ###################################################
    # Internal functions for use with scipy.optimize().
    # These functions will be called directly from scipy.optimize().
    # These functions will take in the problem variables
    # in one column vector np.array with float64 datatype.
    # Each function will the convert the input
    # into the appropiately shaped state and control 
    # variables and convert into a tf.Tensor with a float32 datatype.
    # and call the appropriate objective/constraint functions.
    # They will then cast the result back into a np.array with float64
    ####################################################

    def _scipy_obj(self, variables):
        state, control = self._unflatten_var(variables)
        state, control = self._convert_to_tensor(state, control)
        obj = self.tf_eval_obj(state, control, **self.problem_kwargs)
        obj = self._convert_to_array(obj)
        return obj

    def _scipy_obj_jac(self, variables):
        state, control = self._unflatten_var(variables)
        state, control = self._convert_to_tensor(state, control)
        state_jac, control_jac = self.tf_eval_obj_jac(state, control, **self.problem_kwargs)
        jac = self._flatten_var(state_jac, control_jac)
        jac = self._convert_to_array(jac)
        return jac

    def _scipy_ineq_constraints(self, variables):
        state, control = self._unflatten_var(variables)
        state, control = self._convert_to_tensor(state, control)
        constraints = self.tf_eval_ineq_constraints(state, control, **self.problem_kwargs).numpy().squeeze()

        # negative since scipy considers >= constraints rather than <=
        constraints = -constraints

        constraints = self._convert_to_array(constraints)
        return constraints

    def _scipy_ineq_constraints_jac(self, variables):
        state, control = self._unflatten_var(variables)
        state, control = self._convert_to_tensor(state, control)
        state_jac, control_jac = self.tf_eval_ineq_constraints_jac(state, control, **self.problem_kwargs)

        # reshaping jacobian to be of shape (num_constraints, var_dim)
        state_jac = np.reshape(state_jac, (-1, self.T * self.state_dim))
        control_jac = np.reshape(control_jac, (-1, (self.T-1) * self.control_dim))

        # negative since scipy considers >= constraints rather than <=
        jac = -np.concatenate((state_jac, control_jac), axis=1)
        jac = self._convert_to_array(jac)
        return jac

    def _scipy_eq_constraints(self, variables):
        state, control = self._unflatten_var(variables)
        state, control = self._convert_to_tensor(state, control)
        constraints = self.tf_eval_eq_constraints(state, control, **self.problem_kwargs).numpy().squeeze()

        # negative since scipy considers >= constraints rather than <=
        constraints = -constraints

        constraints = self._convert_to_array(constraints)
        return constraints

    def _scipy_eq_constraints_jac(self, variables):
        state, control = self._unflatten_var(variables)
        state, control = self._convert_to_tensor(state, control)
        state_jac, control_jac = self.tf_eval_eq_constraints_jac(state, control, **self.problem_kwargs)

        # reshaping jacobian to be of shape (num_constraints, var_dim)
        state_jac = np.reshape(state_jac, (-1, self.T * self.state_dim))
        control_jac = np.reshape(control_jac, (-1, (self.T-1) * self.control_dim))

        # negative since scipy considers >= constraints rather than <=
        jac = -np.concatenate((state_jac, control_jac), axis=1)
        jac = self._convert_to_array(jac)
        return jac

    def _scipy_model_ineq_constraints(self, variables):
        state, control = self._unflatten_var(variables)
        state, control = self._convert_to_tensor(state, control)
        constraints = -self._eval_model_ineq_constraints(state, control).numpy().squeeze()
        constraints = self._convert_to_array(constraints)
        return constraints

    def _scipy_model_ineq_constraints_jac(self, variables):
        state, control = self._unflatten_var(variables)
        state, control = self._convert_to_tensor(state, control)
        state_jac, control_jac = self._compute_jac(self._eval_model_ineq_constraints, state, control)

        # reshaping jacobian to be of shape (num_constraints, var_dim)
        state_jac = np.reshape(state_jac, (-1, self.T * self.state_dim))
        control_jac = np.reshape(control_jac, (-1, (self.T-1) * self.control_dim))

        # negative since scipy considers >= constraints rather than <=
        jac = -np.concatenate((state_jac, control_jac), axis=1)
        jac = self._convert_to_array(jac)
        return jac

    def _scipy_model_eq_constraints(self, variables):
        state, control = self._unflatten_var(variables)
        state, control = self._convert_to_tensor(state, control)
        constraints = self.tf_eval_model_eq_constraints(state, control).numpy().squeeze()
        constraints = self._convert_to_array(constraints)
        return constraints

    def _scipy_model_eq_constraints_jac(self, variables):
        state, control = self._unflatten_var(variables)
        state, control = self._convert_to_tensor(state, control)
        state_jac, control_jac = self.tf_eval_model_eq_constraints_jac(state, control)

        # reshaping jacobian to be of shape (num_constraints, var_dim)
        state_jac = np.reshape(state_jac, (-1, self.T * self.state_dim))
        control_jac = np.reshape(control_jac, (-1, (self.T-1) * self.control_dim))
        jac = np.concatenate((state_jac, control_jac), axis=1)
        jac = self._convert_to_array(jac)
        return jac
