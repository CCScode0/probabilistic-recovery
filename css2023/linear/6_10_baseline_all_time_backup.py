from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
from time import perf_counter

from settings_baseline import motor_speed_bias, quadruple_tank_bias, lane_keeping, f16_bias
from utils.formal.gaussian_distribution import GaussianDistribution
from utils.formal.reachability import ReachableSet
from utils.formal.zonotope import Zonotope
from utils.observers.kalman_filter import KalmanFilter
from utils.observers.full_state_bound import Estimator
from utils.controllers.LP_cvxpy import LP
from utils.controllers.MPC_cvxpy import MPC

exps = [quadruple_tank_bias]
# exps = [f16_bias]
# baselines = ['none', 'lp', 'lqr', 'ssr', 'oprp', 'fprp']
baselines = ['none', 'lp', 'lqr', 'ssr', 'oprp']
# baselines = ['none', 'ssr', 'oprp']
colors = {'none': 'red', 'lp': 'cyan', 'lqr': 'blue', 'ssr': 'orange', 'oprp': 'purple', 'fprp': 'violet'}
result = {}  # for print or plot

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

for exp in exps:
    result[exp.name] = {}
    exp_rst = result[exp.name]

    #  =================  no_recovery  ===================
    # if 'none' in baselines:
    if True:
        bl = 'none'
        exp_name = f" none + {exp.name} "
        logger.info(f"{exp_name:^40}")
        for i in range(0, exp.max_index + 1):
            assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            # attack here
            exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
            if i == exp.attack_start_index - 1:
                logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')
            if i == exp.recovery_index:
                logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')
            exp.model.evolve()
        exp_rst[bl] = {}
        exp_rst[bl]['refs'] = deepcopy(exp.model.refs)
        exp_rst[bl]['states'] = deepcopy(exp.model.states)
        exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
        exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
        exp_rst[bl]['time'] = {}
        exp_rst[bl]['time']['recovery_complete'] = exp.max_index-1

    #  =================  LP_recovery  ===================
    exp.model.reset()

    # required objects
    A = exp.model.sysd.A
    B = exp.model.sysd.B
    est = Estimator(A, B, max_k=150, epsilon=1e-7)

    # init variables
    recovery_complete_index = np.inf
    rec_u = None
    recon_t = None
    solve_t = None

    if 'lp' in baselines:
        bl = 'lp'
        exp_name = f" lp + {exp.name} "
        logger.info(f"{exp_name:^40}")
        for i in range(0, exp.max_index + 1):
            assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            # attack here
            exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
            if i == exp.attack_start_index - 1:
                logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')
            if i == exp.recovery_index:
                logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')

                # State reconstruction
                b_recon_t = perf_counter()
                us = exp.model.inputs[exp.attack_start_index - 1:exp.recovery_index]
                x_0 = exp.model.states[exp.attack_start_index - 1]
                x_cur_lo, x_cur_up, x_cur = est.estimate(x_0, us)
                e_recon_t = perf_counter()
                recon_t = e_recon_t - b_recon_t
                logger.debug(f'reconstructed state={x_cur}')

                # deadline estimate
                safe_set_lo = exp.safe_set_lo
                safe_set_up = exp.safe_set_up
                control = exp.model.inputs[i - 1]
                k = est.get_deadline(x_cur, safe_set_lo, safe_set_up, control, 100)
                recovery_complete_index = exp.attack_start_index + k
                logger.debug(f'deadline={k}')

                # get recovery control sequence
                lp_settings = {
                    'Ad': A, 'Bd': B,
                    'N': k,
                    'ddl': k, 'target_lo': exp.target_set_lo, 'target_up': exp.target_set_up,
                    'safe_lo': exp.safe_set_lo, 'safe_up': exp.safe_set_up,
                    'control_lo': exp.control_lo, 'control_up': exp.control_up,
                    'ref': exp.recovery_ref
                }
                lp = LP(lp_settings)
                b_solve_t = perf_counter()
                _ = lp.update(feedback_value=x_cur)
                e_solve_t = perf_counter()
                solve_t = e_solve_t - b_solve_t
                rec_u = lp.get_full_ctrl()
                rec_x = lp.get_last_x()
                logger.debug(f'expected recovery state={rec_x}')

            if exp.recovery_index <= i < recovery_complete_index:
                rec_u_index = i - exp.recovery_index
                u = rec_u[rec_u_index]
                exp.model.evolve(u)
            else:
                if i == recovery_complete_index:
                    logger.debug(f'state after recovery={exp.model.cur_x}')
                    step = recovery_complete_index - exp.recovery_index
                    logger.debug(f'use {step} steps to recover.')
                exp.model.evolve()

        exp_rst[bl] = {}
        exp_rst[bl]['states'] = deepcopy(exp.model.states)
        exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
        exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
        exp_rst[bl]['time'] = {}
        exp_rst[bl]['time']['recovery_complete'] = recovery_complete_index
        exp_rst[bl]['time']['recon'] = recon_t
        exp_rst[bl]['time']['solve'] = solve_t

    #  =================  LQR_recovery  ===================
    # did not add maintainable time estimation, let it to be 3
    maintain_time = 3
    exp.model.reset()

    # init variables
    recovery_complete_index = np.inf
    rec_u = None

    if 'lqr' in baselines:
        bl = 'lqr'
        exp_name = f" lqr + {exp.name} "
        logger.info(f"{exp_name:^40}")
        for i in range(0, exp.max_index + 1):
            assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            # attack here
            exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
            if i == exp.attack_start_index - 1:
                logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')
            if i == exp.recovery_index:
                logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')

                # State reconstruction
                b_recon_t = perf_counter()
                us = exp.model.inputs[exp.attack_start_index - 1:exp.recovery_index]
                x_0 = exp.model.states[exp.attack_start_index - 1]
                x_cur_lo, x_cur_up, x_cur = est.estimate(x_0, us)
                e_recon_t = perf_counter()
                recon_t = e_recon_t - b_recon_t
                logger.debug(f'reconstructed state={x_cur}')

                # deadline estimate
                safe_set_lo = exp.safe_set_lo
                safe_set_up = exp.safe_set_up
                control = exp.model.inputs[i - 1]
                k = est.get_deadline(x_cur, safe_set_lo, safe_set_up, control, 100)
                recovery_complete_index = exp.attack_start_index + k
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
                    'ref': np.array([14, 14, 2, 2.5])
                }
                lqr = MPC(lqr_settings)
                b_solve_t = perf_counter()
                _ = lqr.update(feedback_value=x_cur)
                e_solve_t = perf_counter()
                solve_t = e_solve_t - b_solve_t
                rec_u = lqr.get_full_ctrl()
                rec_x = lqr.get_last_x()
                logger.debug(f'expected recovery state={rec_x}')

            if i == recovery_complete_index:
                logger.debug(f'state after recovery={exp.model.cur_x}')
                step = recovery_complete_index - exp.recovery_index
                logger.debug(f'use {step} steps to recover.')

            if exp.recovery_index <= i < recovery_complete_index + maintain_time:
                rec_u_index = i - exp.recovery_index
                u = rec_u[rec_u_index]
                exp.model.evolve(u)
            else:
                exp.model.evolve()

        exp_rst[bl] = {}
        exp_rst[bl]['states'] = deepcopy(exp.model.states)
        exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
        exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
        exp_rst[bl]['time'] = {}
        exp_rst[bl]['time']['recovery_complete'] = recovery_complete_index + maintain_time
        exp_rst[bl]['time']['recon'] = recon_t
        exp_rst[bl]['time']['solve'] = solve_t

    #  =================  Software_sensor_recovery  ===================
    exp.model.reset()

    # required objects
    def in_target_set(target_lo, target_hi, x_cur):
        res = True
        for i in range(len(x_cur)):
            if target_lo[i] > x_cur[i] or target_hi[i] < x_cur[i]:
                res = False
                break
        return res

    # init variables
    recovery_complete_index = np.inf
    last_predicted_state = None

    if 'ssr' in baselines:
        bl = 'ssr'
        exp_name = f" ssr + {exp.name} "
        logger.info(f"{exp_name:^40}")
        for i in range(0, exp.max_index + 1):
            assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            # attack here
            exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
            if i == exp.attack_start_index - 1:
                logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')
            if i == exp.recovery_index:
                logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')

                # State reconstruction
                b_recon_t = perf_counter()
                us = exp.model.inputs[exp.attack_start_index - 1:exp.recovery_index-1]
                x_0 = exp.model.states[exp.attack_start_index - 1]
                x_cur = est.estimate_wo_bound(x_0, us)
                e_recon_t = perf_counter()
                recon_t = e_recon_t - b_recon_t
                logger.debug(f'one before reconstructed state={x_cur}')
                last_predicted_state = deepcopy(x_cur)

            if exp.recovery_index <= i <= recovery_complete_index:
                # check if it is in target set
                # if in_target_set(exp.target_set_lo, exp.target_set_up, last_predicted_state):
                #     recovery_complete_index = i
                #     logger.debug('state after recovery={exp.model.cur_x}')
                #     step = recovery_complete_index - exp.recovery_index
                #     logger.debug(f'use {step} steps to recover.')
                us = [exp.model.inputs[i - 1]]
                x_0 = last_predicted_state
                x_cur = est.estimate_wo_bound(x_0, us)
                exp.model.cur_feedback = exp.model.sysd.C @ x_cur
                last_predicted_state = deepcopy(x_cur)
                print(f'{exp.model.cur_u}')
            exp.model.evolve()

        exp_rst[bl] = {}
        exp_rst[bl]['states'] = deepcopy(exp.model.states)
        exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
        exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
        exp_rst[bl]['time'] = {}
        exp_rst[bl]['time']['recovery_complete'] = exp.max_index-1
        exp_rst[bl]['time']['recon'] = recon_t
        exp_rst[bl]['time']['solve'] = -1
        # print(f'{recovery_complete_index}')

    #  =================  Optimal_probabilistic_recovery  ===================
    exp.model.reset()

    # required objects
    kf_C = exp.kf_C
    C = exp.model.sysd.C
    D = exp.model.sysd.D
    kf_Q = exp.model.p_noise_dist.sigma if exp.model.p_noise_dist is not None else np.zeros_like(A)
    kf_R = exp.kf_R
    kf = KalmanFilter(A, B, kf_C, D, kf_Q, kf_R)
    U = Zonotope.from_box(exp.control_lo, exp.control_up)
    W = exp.model.p_noise_dist
    reach = ReachableSet(A, B, U, W, max_step=exp.max_recovery_step + 2)

    # init variables
    recovery_complete_index = np.inf
    x_cur_update = None

    if 'oprp' in baselines:
        bl = 'oprp'
        exp_name = f" oprp + {exp.name} "
        logger.info(f"{exp_name:^40}")
        for i in range(0, exp.max_index + 1):
            assert exp.model.cur_index == i
            exp.model.update_current_ref(exp.ref[i])
            # attack here
            exp.model.cur_feedback = exp.attack.launch(exp.model.cur_feedback, i, exp.model.states)
            if i == exp.attack_start_index - 1:
                logger.debug(f'trustworthy_index={i}, trustworthy_state={exp.model.cur_x}')

            # state reconstruct
            if i == exp.recovery_index:
                logger.debug(f'recovery_index={i}, recovery_start_state={exp.model.cur_x}')
                b_recon_t = perf_counter()
                us = exp.model.inputs[exp.attack_start_index-1:exp.recovery_index]
                ys = (kf_C @ exp.model.states[exp.attack_start_index:exp.recovery_index + 1].T).T
                x_0 = exp.model.states[exp.attack_start_index-1]
                x_res, P_res = kf.multi_steps(x_0, np.zeros_like(A), us, ys)
                x_cur_update = GaussianDistribution(x_res[-1], P_res[-1])
                e_recon_t = perf_counter()
                recon_t = e_recon_t - b_recon_t
                logger.debug(f"reconstructed state={x_cur_update.miu=}, ground_truth={exp.model.cur_x}")
                # x_cur_update = GaussianDistribution(exp.model.cur_x, P_res[-1])

            if exp.recovery_index < i < recovery_complete_index:
                x_cur_predict = GaussianDistribution(*kf.predict(x_cur_update.miu, x_cur_update.sigma, exp.model.cur_u))
                y = kf_C @ exp.model.cur_x
                x_cur_update = GaussianDistribution(*kf.update(x_cur_predict.miu, x_cur_predict.sigma, y))
                logger.debug(f"reconstructed state={x_cur_update.miu=}, ground_truth={exp.model.cur_x}")

            if i == recovery_complete_index:
                logger.debug(f'state after recovery={exp.model.cur_x}')

            if exp.recovery_index <= i < recovery_complete_index:
                reach.init(x_cur_update, exp.s)
                b_solve_t = perf_counter()
                k, X_k, D_k, z_star, alpha, P, arrive = reach.given_k(max_k=exp.max_recovery_step)
                e_solve_t = perf_counter()
                solve_t = e_solve_t - b_solve_t
                print(f"{k=}, {z_star=}, {P=}")
                recovery_control_sequence = U.alpha_to_control(alpha)
                recovery_complete_index = i+k

                exp.model.evolve(recovery_control_sequence[0])
                print(f"{i=}, {recovery_control_sequence[0]=}")
            else:
                exp.model.evolve()




        exp_rst[bl] = {}
        exp_rst[bl]['states'] = deepcopy(exp.model.states)
        exp_rst[bl]['outputs'] = deepcopy(exp.model.outputs)
        exp_rst[bl]['inputs'] = deepcopy(exp.model.inputs)
        exp_rst[bl]['time'] = {}
        exp_rst[bl]['time']['recovery_complete'] = recovery_complete_index
        exp_rst[bl]['time']['recon'] = recon_t
        exp_rst[bl]['time']['solve'] = solve_t


    # ==================== plot =============================
    plt.rcParams.update({'font.size': 18})  # front size
    fig = plt.figure(figsize=(8, 4))

    # plot reference
    t_arr = np.linspace(0, exp.dt * exp.max_index, exp.max_index + 1)[:exp.max_index]
    ref = [x[exp.ref_index] for x in exp_rst['none']['refs'][:exp.max_index]]
    plt.plot(t_arr, ref, color='grey', linestyle='dashed')
    # plot common part (before recovery)
    t_arr_common = t_arr[:exp.recovery_index + 1]
    output = [x[exp.output_index] for x in exp_rst['none']['outputs'][:exp.recovery_index + 1]]
    plt.plot(t_arr_common, output, color='black')
    # plot attack / recovery

    for bl in baselines:
        end_time = exp_rst[bl]['time']['recovery_complete']
        t_arr_tmp = t_arr[exp.recovery_index:end_time+1]
        output = [x[exp.output_index] for x in exp_rst[bl]['outputs'][exp.recovery_index:end_time+1]]
        plt.plot(t_arr_tmp, output, color=colors[bl], label=bl)

    plt.ylim(exp.y_lim)
    plt.xlim(exp.x_lim, exp.max_index*exp.dt)

    plt.legend()
    plt.show()


# time print
headers = ['exp_name', 'baseline', 'reconstr', 'prepare', 'control']
for exp_rst_name in result:
    logger.info(f'{exp_rst_name:=^40}')
    exp_rst = result[exp_rst_name]
    for bl in baselines:
        if bl == 'none':
            continue
        logger.info(f'{bl:=^30}')
        logger.info(f"reconstruction time {exp_rst[bl]['time']['recon']}")
        logger.info(f"solving time        {exp_rst[bl]['time']['solve']}")