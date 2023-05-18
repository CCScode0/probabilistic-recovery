from copy import deepcopy
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import logging
import sys
import os
import csv
import time
from time import perf_counter

os.environ["RANDOM_SEED"] = '0'   # for reproducibility


from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.reachability import ReachableSet
from utils.formal.zonotope import Zonotope
from utils.observers.kalman_filter import KalmanFilter
from utils.observers.full_state_bound import Estimator
from utils.controllers.LP_cvxpy import LP
from utils.controllers.MPC_cvxpy import MPC

from utils.performance.performance_metrics import distance_to_strip_center, in_strip



colors = {'none': 'red', 'lqr': 'C1', 'ssr': 'C2', 'oprp-close': 'C4', 'oprp-open': 'C0'}
labels = {'none': 'None', 'lqr': 'RTR-LQR', 'ssr': 'VS', 'oprp-close': 'OPRP-CL', 'oprp-open': 'OPRP'}

# logger
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


parallel = False
plot_time_series = True



def save_exp_data(exp, exp_rst, bl, profiling_data = None):
    '''method to store data'''
    exp_rst[bl]['states'] = deepcopy(exp.model.states)
    exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
    exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
    exp_rst[bl]['time'] = {}

    exp_rst[bl]['recovery_steps'] = profiling_data['recovery_steps']
    exp_rst[bl]['recovery_complete_index'] = profiling_data['recovery_complete_index']
    
        

# ---------  attack + no recovery  -------------
def simulate_no_recovery(exp, bl):
    exp.model.reset()
    exp_name = f" {bl} {exp.name} "
    # logger.info(f"{exp_name:=^40}")
    for i in range(0, exp.max_index + 1):
        assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])
        # attack here
        exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
        exp.model.evolve()

    profiling_data = {}
    profiling_data["recovery_complete_index"] = exp.max_index + 1
    profiling_data["recovery_steps"] = exp.max_index - exp.attack_start_index 
    
    return profiling_data


# ---------  LQR  -------------
def simulate_LQR(exp, bl):
    exp_name = f" {bl} {exp.name} "
    logger.info(f"{exp_name:=^40}")
    # required objects
    A = exp.model.sysd.A
    B = exp.model.sysd.B
    est = Estimator(A, B, max_k=150, epsilon=1e-7)
    maintain_time = 3
    exp.model.reset()
    # init variables
    recovery_complete_index = exp.max_index
    rec_u = None

    elapsed_times = []
    for i in range(0, exp.max_index + 1):
        assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])
        # attack here
        exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
        if i == exp.attack_start_index - 1:
            logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')
            pass

        

        if i == exp.recovery_index:
            logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')

            # State reconstruction
            us = exp.model.inputs[exp.attack_start_index - 1:exp.recovery_index]
            x_0 = exp.model.states[exp.attack_start_index - 1]
            x_cur_lo, x_cur_up, x_cur = est.estimate(x_0, us)
            logger.debug(f'reconstructed state={x_cur}')

            # deadline estimate
            safe_set_lo = exp.safe_set_lo
            safe_set_up = exp.safe_set_up
            control = exp.model.inputs[i - 1]
            k = est.get_deadline(x_cur, safe_set_lo, safe_set_up, control, 100)
            recovery_complete_index = exp.recovery_index + k
            logger.debug(f'deadline={k}')
            # maintainable time compute


            # get recovery control sequence
            lqr_settings = {
                'Ad': A, 'Bd': B,
                'Q': exp.Q, 'QN': exp.QN, 'R': exp.R,
                'N': k + 3,
                'ddl': k, 'target_lo': exp.target_set_lo, 'target_up': exp.target_set_up,
                'safe_lo': exp.safe_set_lo, 'safe_up': exp.safe_set_up,
                'control_lo': exp.control_lo, 'control_up': exp.control_up,
                'ref': exp.recovery_ref
            }
            lqr = MPC(lqr_settings)
            _ = lqr.update(feedback_value=x_cur)
            rec_u = lqr.get_full_ctrl()
            rec_x = lqr.get_last_x()
            logger.debug(f'expected recovery state={rec_x}')

        if i == recovery_complete_index + maintain_time:
            logger.debug(f'state after recovery={exp.model.cur_x}')
            step = recovery_complete_index + maintain_time - exp.recovery_index
            logger.debug(f'use {step} steps to recover.')

        if exp.recovery_index <= i < recovery_complete_index + maintain_time:
            rec_u_index = i - exp.recovery_index
            u = rec_u[rec_u_index]

            

            exp.model.evolve(u)
        else:
            exp.model.evolve()
    
    profiling_data = {}
    profiling_data["recovery_complete_index"] = recovery_complete_index
    profiling_data["recovery_steps"] = step
    return profiling_data


# ---------  attack + virtual sensors  -------------
def simulate_ssr(exp, bl):
    # required objects
    recovery_complete_index = exp.max_index - 1
    A = exp.model.sysd.A
    B = exp.model.sysd.B
    est = Estimator(A, B, max_k=150, epsilon=1e-7)
    logger.info(f"{bl} {exp.name:=^40}")
    for i in range(0, exp.max_index + 1):
        assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])
        # attack here
        exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)

        if i == exp.attack_start_index - 1:
            logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')
            pass
        if i == exp.recovery_index:
            logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')

            # State reconstruction
            us = exp.model.inputs[exp.attack_start_index - 1:exp.recovery_index]
            x_0 = exp.model.states[exp.attack_start_index - 1]
            x_cur = est.estimate_wo_bound(x_0, us)
            logger.debug(f'one before reconstructed state={x_cur}')
            last_predicted_state = deepcopy(x_cur)

        if exp.recovery_index <= i < recovery_complete_index:
            # check if it is in target set
            if in_strip(exp.s.l, last_predicted_state, exp.s.a, exp.s.b):
                recovery_complete_index = i
                logger.debug('state after recovery={exp.model.cur_x}')
                step = recovery_complete_index - exp.recovery_index
                logger.debug(f'use {step} steps to recover.')
            us = [exp.model.inputs[i - 1]]
            x_0 = last_predicted_state
            x_cur = est.estimate_wo_bound(x_0, us)
            exp.model.cur_feedback = exp.model.sysd.C @ x_cur
            last_predicted_state = deepcopy(x_cur)
            # print(f'{exp.model.cur_u}')


        exp.model.evolve()
    
    profiling_data = {}
    profiling_data["recovery_complete_index"] = recovery_complete_index
    profiling_data["recovery_steps"] = recovery_complete_index - exp.recovery_index 
    return profiling_data


# ---------  attack + OPRP  -------------
def simulate_oprp_cl(exp, bl):
    # required objects
    A = exp.model.sysd.A
    B = exp.model.sysd.B
    kf_C = exp.kf_C
    C = exp.model.sysd.C
    D = exp.model.sysd.D
    kf_Q = exp.model.p_noise_dist.sigma if exp.model.p_noise_dist is not None else np.zeros_like(A)
    kf_R = exp.kf_R
    kf = KalmanFilter(A, B, kf_C, D, kf_Q, kf_R)
    U = Zonotope.from_box(exp.control_lo, exp.control_up)
    W = exp.model.p_noise_dist
    reach = ReachableSet(A, B, U, W, max_step=exp.max_recovery_step )

    # init variables
    recovery_complete_index = exp.max_index
    x_cur_update = None

    exp_name = f" {bl} {exp.name} "
    

    for i in range(0, exp.max_index + 1):
        assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])
        # attack here
        exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)

        if i == exp.attack_start_index - 1:
            logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')
            pass

        # state reconstruct
        if i == exp.recovery_index:
            logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')
            us = exp.model.inputs[exp.attack_start_index-1:exp.recovery_index]
            ys = (kf_C @ exp.model.states[exp.attack_start_index:exp.recovery_index + 1].T).T
            x_0 = exp.model.states[exp.attack_start_index-1]
            x_res, P_res = kf.multi_steps(x_0, np.zeros_like(A), us, ys)
            x_cur_update = GaussianDistribution(x_res[-1], P_res[-1])
            logger.debug(f"reconstructed state={x_cur_update.miu=}, ground_truth={exp.model.cur_x}")

        if exp.recovery_index < i < recovery_complete_index:
            x_cur_predict = GaussianDistribution(*kf.predict(x_cur_update.miu, x_cur_update.sigma, exp.model.cur_u))
            y = kf_C @ exp.model.cur_x
            x_cur_update = GaussianDistribution(*kf.update(x_cur_predict.miu, x_cur_predict.sigma, y))
            logger.debug(f"reconstructed state={x_cur_update.miu=}, ground_truth={exp.model.cur_x}")

        if i == recovery_complete_index:
            logger.debug(f'state after recovery={exp.model.cur_x}')
            pass

        if exp.recovery_index <= i < recovery_complete_index:
            reach.init(x_cur_update, exp.s)
            k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=exp.max_recovery_step)
            # print(f"{k=}, {z_star=}, {P=}")
            recovery_control_sequence = U.alpha_to_control(alpha)
            recovery_complete_index = i+k
            exp.model.evolve(recovery_control_sequence[0])
            # print(f"{i=}, {recovery_control_sequence[0]=}")
        else:
            exp.model.evolve()
    
    profiling_data = {}
    profiling_data["recovery_complete_index"] = recovery_complete_index
    profiling_data["recovery_steps"] = recovery_complete_index - exp.recovery_index 
    return profiling_data

# ---------  attack + OPRP_CL  -------------
def simulate_oprp_ol(exp, bl):
    # required objects
    A = exp.model.sysd.A
    B = exp.model.sysd.B
    C = exp.model.sysd.C
    U = Zonotope.from_box(exp.control_lo, exp.control_up)
    W = exp.model.p_noise_dist
    reach = ReachableSet(A, B, U, W, max_step=exp.max_recovery_step )

    # init variables
    recovery_complete_index = exp.max_index
    x_cur_update = None
    exp_name = f" {bl} {exp.name} "
    logger.info(f"{exp_name:=^40}")
    for i in range(0, exp.max_index + 1):
        assert exp.model.cur_index == i
        exp.model.update_current_ref(exp.ref[i])
        # attack here
        exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
        if i == exp.attack_start_index - 1:
            logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')
            pass

        # state reconstruct
        if i == exp.recovery_index:
            us = exp.model.inputs[exp.attack_start_index - 1:exp.recovery_index]
            x_0 = exp.model.states[exp.attack_start_index - 1]
            x_0 = GaussianDistribution(x_0, np.zeros((exp.model.n, exp.model.n)))
            reach.init(x_0, exp.s)
            x_res_point = reach.state_reconstruction(us)
            print('x_0=', x_res_point)

            reach.init(x_res_point, exp.s)
            k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=exp.k_given)
            recovery_complete_index = exp.recovery_index + k
            rec_u = U.alpha_to_control(alpha)
            
            print('k=', k, 'P=', P, 'z_star=', z_star, 'arrive=', arrive)
            print('D_k=', D_k)
            print('recovery_complete_index=', recovery_complete_index)
            print(rec_u)

        if exp.recovery_index <= i < recovery_complete_index:
            rec_u_index = i - exp.recovery_index
            u = rec_u[rec_u_index]
            exp.model.evolve(u)
        else:
            exp.model.evolve()
    
    profiling_data = {}
    profiling_data["recovery_complete_index"] = recovery_complete_index
    profiling_data["recovery_steps"] = recovery_complete_index - exp.recovery_index 

    return profiling_data


def sim_strategies(exps):
    
    for exp in exps:

        strategies = {}
        strategies = {  # 'none': simulate_no_recovery,
                        'ssr': simulate_ssr,
                        'oprp-close': simulate_oprp_cl,
                        'oprp-open': simulate_oprp_ol,
                        'lqr': simulate_LQR}
        
        
        result = {}
        result[exp.name] = {}
        exp_rst = result[exp.name]
        for strategy in strategies:
            exp.model.reset()
            exp_rst[strategy] = {}
            simulate = strategies[strategy]
            profiling_data = simulate(exp, strategy)
            save_exp_data(exp, exp_rst, strategy, profiling_data)


            recovery_complete_index = exp_rst[strategy]['recovery_complete_index']
            final_state = exp_rst[strategy]['states'][recovery_complete_index].tolist()
            

            
        # Begin plots here
        if plot_time_series:
            plt.figure(figsize=(5,3))
            max_t = 0
            for strategy in strategies:
                # 
                states = exp_rst[strategy]["states"]
                t = np.arange(0, len(states)) * exp.dt

                states = states[0:exp_rst[strategy]["recovery_complete_index"] + 1]
                t = t[0:exp_rst[strategy]["recovery_complete_index"] + 1]
                max_t = np.maximum(t[-1], max_t)
                #
                plt.plot(t, states[:, 0], color=colors[strategy], label=labels[strategy], linewidth=2)
                plt.plot(t[-1], states[-1, 0], color=colors[strategy], marker='*', linewidth=8)
            # Plot limits in x
            axes = plt.gca()
            x_lim = axes.get_xlim()
            plt.xlim(exp.x_lim[0], max_t + 0.5)

            # Plot limits in y
            plt.ylim(exp.y_lim)
            plt.vlines(exp.attack_start_index * exp.dt, exp.y_lim[0], exp.y_lim[1], color='red', linestyle='dashed', linewidth=2)
            plt.vlines(exp.recovery_index * exp.dt, exp.y_lim[0], exp.y_lim[1], colors='green', linestyle='dotted', linewidth=2)

            plt.fill_between(t + 0.5, t * 0 - exp.s.a, t * 0 - exp.s.b, facecolor='green', alpha=0.3)
            plt.ylabel("Rotational speed [rad/s]")
            plt.xlabel("Time [s]")
            plt.legend(ncol=2)
            plt.show()

def run_once():
    
    from ccs2023.linear.settings_baseline import motor_speed_bias, quadruple_tank_bias, f16_bias, aircraft_pitch_bias, boeing747_bias, quadrotor_bias, rlc_circuit_bias
    rseed = np.uint32(int(time.time()))
    exps = [motor_speed_bias]
    init = sim_strategies(exps)


def main():
    headers = ['benchmark', 'TS', 'RTR-LQR', 'VS', 'OPRP']
    times = 100
    run_once()

    
if __name__ == "__main__":
    main()
    
        

