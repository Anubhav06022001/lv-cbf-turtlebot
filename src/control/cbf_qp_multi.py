import osqp
import numpy as np
import scipy.sparse as sp

def cbf_qp_multi(u_ref, A_list, b_list):
    n = len(u_ref) 

    P_diag = np.ones(n + 1) * 2.0
    P_diag[-1] = 1000.0  
    P = sp.diags(P_diag).tocsc()
    
    q = np.zeros(n + 1)
    q[:n] = -2 * u_ref

    if not A_list: return u_ref

    A_cbf = np.vstack(A_list)
    b_cbf = np.array(b_list)         

    # u_min, u_max = -2.0 * np.ones(n), 2.0 * np.ones(n)
    u_min = np.array([0, -2.75, 0.0])  
    u_max = np.array([0.21, +2.75, 0.0])

    # Add slack column to CBF constraints
    slack_col = np.ones((len(b_cbf), 1))
    A_cbf_slack = np.hstack([A_cbf, slack_col])
    
    # Velocity Bounds
    A_bound = np.hstack([np.eye(n), np.zeros((n, 1))])
    
    A_full = np.vstack([A_cbf_slack, A_bound])
    l_full = np.concatenate([b_cbf, u_min])
    u_full = np.concatenate([np.inf * np.ones(len(b_cbf)), u_max])

    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=sp.csc_matrix(A_full), l=l_full, u=u_full, verbose=False)
    res = solver.solve()
    
    if res.info.status != 'solved':
        print(f"[Warning] QP Infeasible! Status: {res.info.status}")
        return u_ref 
        
    return res.x[:n]