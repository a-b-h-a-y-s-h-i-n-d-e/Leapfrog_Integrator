import torch
from torch.autograd import grad
from tqdm import tqdm

def leapfrog(q0: torch.Tensor,
             p0: torch.Tensor,
             Func: callable,
             dt: float,
             N: int,
             is_hamiltonian: bool =True) -> torch.Tensor: 
    
    # (q, p) occurence
    trajectories = torch.empty((N, q0.shape[0] + p0.shape[0]))

    p = p0
    q = q0
    p.requires_grad_()
    q.requires_grad_()

    if is_hamiltonian:
        hamilt = Func(torch.cat([q, p], dim=-1))
        dpdt = -grad(hamilt.sum(), q, create_graph=True)[0]

        for i in range(N):

            p_half = p + (dt / 2) * dpdt

            trajectories[i, :q0.shape[0]] = q
            trajectories[i, q0.shape[0]:] = p

            hamil = Func(torch.cat([q, p_half], dim=-1))
            dqdt = grad(hamil.sum(), p_half, create_graph=True)[0]
            q_next = q + dt * dqdt

            hamil = Func(torch.cat([q_next, p_half], dim=-1))
            dpdt = -grad(hamil.sum(), q_next, create_graph=True)[0]
            p_next = p_half + (dt / 2) * dpdt

            q, p = q_next, p_next
    else: 
        time_drvt = Func((q, p))
        dpdt = time_drvt[0]

        for i in tqdm(range(N)):

            p_half = p + dpdt * (dt / 2)

            trajectories[i, :q0.shape[0]] = q
            trajectories[i, q0.shape[0]:] = p

            time_drvt = Func((q, p_half))
            dqdt = time_drvt[1]
            q_next = q + dqdt * dt

            time_drvt = Func((q_next, p_half))
            dpdt = time_drvt[0]
            p_next = p_half + dpdt * (dt / 2)

            q, p = q_next, p_next

    # shape - (N, (q, p))
    return trajectories
