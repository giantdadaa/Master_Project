import numpy as np
from scipy.optimize import minimize


def skew_sym_regress(X, X_dot, tol=1e-4):
  """
  Original data tensor is C x L x N where N is number of Neurons, L is length of each trial
  and C is number of conditions. We stack this to get L*C x N array.

  Args
  ----
    X_dot: First difference of (reduced dimension) data. Shape is T x N
           
    X: reduced dimension data. Shape is T x N
  """
     
  # 1) Initialize h using the odd part of the least-squares solution.
  # 2) call scipy.optimize.minimize and pass in our starting h, and x_dot, 
  T, N = X.shape
  M_lstq, _, _, _ = np.linalg.lstsq(X, X_dot, rcond=None)    # solves X @ M_lstq = X_dot, unconstrained least squares solution
  M_lstq = M_lstq.T
  M_init = 0.5 * (M_lstq - M_lstq.T)                         # extracts only the skew-symmetric part
  h_init = _reshape_mat2vec(M_init, N)

  options=dict(maxiter=10000, gtol=tol)
  result = minimize(lambda h: _objective(h, X, X_dot),       # minimize the objective function (error between predicted and actual derivatives)
                    h_init,
                    jac=lambda h: _grad_f(h, X, X_dot),
                    method='CG',                             # Conjugate Gradient method
                    options=options)
  if not result.success:
    print("Optimization failed.")
    print(result.message)
  M = _reshape_vec2mat(result.x, N)
  assert(np.allclose(M, -M.T))     # Verify skew-symmetry
  return M


def _grad_f(h, X, X_dot):
  _, N = X.shape
  M = _reshape_vec2mat(h, N)
  dM = (X.T @ X @ M.T) - X.T @ X_dot            # computes the unconstrained gradient
  return _reshape_mat2vec(dM.T - dM, N)         # projects it onto the skew-symmetric subspace


def _objective(h, X, X_dot):
  '''
  this objective measures the sum of squared errors between the predicted derivative (X·Mᵀ) and
  the actual derivative (Ẋ). The ½ factor is just for mathematical convenience (it cancels when
  taking derivatives).
  '''
  _, N = X.shape
  M = _reshape_vec2mat(h, N)
  return 0.5 * np.linalg.norm(X @ M.T - X_dot, ord='fro')**2


def _reshape_vec2mat(h, N):
  '''Convert parameter vector to skew-symmetric matrix'''
  M = np.zeros((N, N))
  upper_tri_indices = np.triu_indices(N, k=1)
  M[upper_tri_indices] = h
  return M - M.T


def _reshape_mat2vec(M, N):
  '''Convert skew-symmetric matrix to parameter vector'''
  upper_tri_indices = np.triu_indices(N, k=1)
  return M[upper_tri_indices]