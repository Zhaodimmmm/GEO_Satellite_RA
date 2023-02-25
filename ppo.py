# -*-coding:utf-8-*-
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import time
import core
import json
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
# 实时动态图
import matplotlib.pyplot as plt
import satellite_run
from SINR_Calculate import *
from sklearn import preprocessing

ax = []  # 定义一个 x 轴的空列表用来接收动态的数据
ay = []  # 定义一个 y 轴的空列表用来接收动态的数据
plt.ion()  # 开启一个画图的窗口


class PPOBuffer:
    def __init__(self, obs_dim, rbg_dim, inv_dim, max_req, act_dim, size, gamma=0.99, lam=0.95):
        self.max_req = max_req
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.rbg_buf = np.zeros(core.combined_shape(size, rbg_dim), dtype=np.int)  # rbg占用情况
        self.inv_buf = np.zeros(core.combined_shape(size, inv_dim), dtype=np.int)  # 无效请求标志位
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.tti2ptr = {}
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store_pending(self, tti, obs, act, val, logp):
        self.tti2ptr[tti] = (obs, act, val, logp)

    def reward_throughput(self, capacity, txBytes, rbg_usable, rbg_used):
        if capacity == 0:
            return 0
        else:
            x1 = txBytes / capacity
            if x1 == 1:
                print(txBytes)
                print(capacity)
        return x1

    def reward_fairness(self, unassign, req_ue):
        if unassign == 0:
            return 1
        else:
            if req_ue - unassign == 0:
                return 0
            else:
                return (req_ue - unassign) / req_ue

    def get_reward(self, capacity, txBytes, rbg_usable, rbg_used, req, unassign, rbg_needed):
        r1 = self.reward_fairness(unassign, req)
        r2 = min(rbg_used, rbg_usable) / rbg_usable
        r3 = self.reward_throughput(capacity, txBytes, rbg_usable, rbg_used)
        r = r3
        return r, r1, r2

    def execute_pop(self, tti, tti_reward):
        try:
            (obs, act, val, logp) = self.tti2ptr.pop(tti)  # 删除该tti，返回的是删除的tti的信息
            ooo = np.array(obs['Requests'])
            rbg_need = np.sum(ooo.reshape(self.max_req, -1)[:, 7])
            if rbg_need == 0:
                return self.store(obs, act, val, logp, 0)
            else:
                return self.store(obs, act, val, logp, tti_reward)
        except KeyError as e:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!KeyError", e)
            return False

    def store(self, obs, act, val, logp, reward):
        if self.ptr < self.max_size:  # buffer has to have room so you can store
            self.obs_buf[self.ptr] = obs['Requests']
            self.rbg_buf[self.ptr] = obs['RbgMap']
            self.inv_buf[self.ptr] = obs['InvFlag']
            self.act_buf[self.ptr] = act
            self.val_buf[self.ptr] = val
            self.logp_buf[self.ptr] = logp
            self.rew_buf[self.ptr] = reward
            self.ptr += 1
            return True
        else:
            raise IndexError()
            return False

    def finish_path(self, last_val=0, denorm=None):
        self.tti2ptr.clear()
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def __len__(self):
        return self.ptr

    def get(self):
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, rbg=self.rbg_buf, inv=self.inv_buf,
                    act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def ppo(env_fn, actor_critic=core.RA_ActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=200, epochs=1000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=200,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10, use_cuda=True):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn.Env()

    use_cuda = use_cuda and torch.cuda.is_available()
    print('use_cuda', use_cuda)

    # assert isinstance(env.observation_space, gym.spaces.Dict)
    max_req = env.observation_space["Requests"][0]
    obs_dim = np.prod(env.observation_space["Requests"])
    rbg_dim = np.prod(env.observation_space["RbgMap"])
    inv_dim = np.prod(env.observation_space["InvFlag"])
    act_dim = env.action_space[0]

    # Create actor-critic module
    ac_kwargs['use_cuda'] = use_cuda
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())  # 即steps_per_epoch
    buf = PPOBuffer(obs_dim, rbg_dim, inv_dim, max_req, act_dim, local_steps_per_epoch, gamma, lam)

    def entropy(dist):
        min_real = torch.finfo(dist.logits.dtype).min
        logits = torch.clamp(dist.logits, min=min_real)
        p_log_p = logits * dist.probs
        return -p_log_p.sum(-1)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, rbg, inv, act, adv, logp_old = data['obs'], data['rbg'], data['inv'], data['act'], data['adv'], data['logp']
        if use_cuda:
            obs, rbg, inv, act, adv, logp_old = obs.cuda(), rbg.cuda(), inv.cuda(), act.cuda(), adv.cuda(), logp_old.cuda()
        pi, logp = ac.pi(obs, rbg, inv, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = entropy(pi).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, rbg, inv, ret = data['obs'].float(), data['rbg'].float(), data['inv'].float(), data['ret'].float()
        if use_cuda:
            obs, rbg, inv, ret = obs.cuda(), rbg.cuda(), inv.cuda(), ret.cuda()
        inp = torch.cat((obs, rbg, inv), dim=1)
        return ((ac.v(inp) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()
        ac.train()
        # Value function learning
        v_l_old = None
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            if v_l_old is None:
                v_l_old = loss_v.item()
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()
        pi_l_old, pi_info_old = None, None
        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            if pi_l_old is None:
                pi_l_old, pi_info_old = loss_pi.item(), pi_info
            kl = mpi_avg(pi_info['kl'])
            if kl > 50 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()
        ac.eval()
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

    # Prepare for interaction with environment
    ontime = 20
    offtime = 2
    _, o = env.reset(ontime, offtime)
    fig, ax = plt.subplots()
    x = []
    y = []
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        pbar = tqdm(total=local_steps_per_epoch)
        # upper_per_rbg = setting.rbgcapa
        ac.eval()
        ep_tx, ep_capacity, ep_waiting, ep_ret, ep_len, ep_newbytes, ep_bler, ep_rbg_used = 0, 0, 0, 0, 0, 0, 0, 0
        epoch_tx, epoch_capacity, epoch_waiting, epoch_reward, epoch_newbytes, epoch_bler, epoch_rbg_used = 0, 0, 0, 0, 0, 0, 0
        ep_r1, ep_r2 = 0, 0
        sum_tx = 0
        error = 0
        final_waiting = 0

        while len(buf) < local_steps_per_epoch:
            obs = torch.as_tensor(o["Requests"], dtype=torch.float32)
            rbg = torch.as_tensor(o["RbgMap"], dtype=torch.int32)
            fla = torch.as_tensor(o["InvFlag"], dtype=torch.int32)
            a, v, logp = ac.step(obs, rbg, fla)
            info, next_o, extra, d = env.step(obs, a)
            capa_list = get_capacity(env.beam_num)
            # 当前的一个obs_tti
            tti_reward = 0
            tti_r1, tti_r2 = 0, 0
            cell_reward_list = []
            waiting_bytes, tx_bytes, capacity_bytes, new_bytes, bler, rbg_used = 0, 0, 0, 0, 0, 0
            for _, cell in extra.items():
                tx_bytes += cell['last_time_txdata']
                new_bytes += cell['newdata']
                rbg_used += cell['rbg_used']
                capacity = min(int(cell['waitingdata']), capa_list[cell['enb'] - 1])
                rrr, r_bler_rbg, r_req_unalloca = buf.get_reward(capacity, int(cell['last_time_txdata']),
                                                                 int(cell['rbg_usable']),
                                                                 int(cell['rbg_used']), int(cell['enb_req_total']),
                                                                 int(cell['unassigned_total']),
                                                                 int(cell['number_of_rbg_nedded']), )
                cell_reward_list.append(rrr)
                tti_reward += rrr
                tti_r1 += r_bler_rbg
                tti_r2 += r_req_unalloca
            tti_reward = tti_reward / len(cell_reward_list) if len(cell_reward_list) != 0 else 0
            tti_r1 = tti_r1 / len(cell_reward_list) if len(cell_reward_list) != 0 else 0
            tti_r2 = tti_r2 / len(cell_reward_list) if len(cell_reward_list) != 0 else 0
            ep_r1 += tti_r1
            ep_r2 += tti_r2
            ep_tx += tx_bytes
            ep_capacity += capacity_bytes
            ep_newbytes += new_bytes
            ep_rbg_used += rbg_used
            ep_bler += bler
            ep_ret += tti_reward
            ep_len += 1
            buf.store(o, a, v, logp, tti_reward)
            pbar.update(1)
            o = next_o
            timeout = ep_len == max_ep_len  # 一个episode
            terminal = timeout or d
            epoch_ended = len(buf) == local_steps_per_epoch  # 一个epoch
            # 缓存满了，触发更新
            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    obs = torch.as_tensor(o["Requests"], dtype=torch.float32)
                    rbg = torch.as_tensor(o["RbgMap"], dtype=torch.int32)
                    fla = torch.as_tensor(o["InvFlag"], dtype=torch.int32)
                    _, v, _ = ac.step(obs, rbg, fla)
                else:
                    v = 0
                buf.finish_path(v)  # 一个episode
                epoch_reward += ep_ret
                epoch_capacity += ep_capacity
                epoch_tx += ep_tx
                epoch_newbytes += ep_newbytes
                epoch_bler += ep_bler
                epoch_rbg_used += ep_rbg_used
                ep_tx, ep_capacity, ep_waiting, ep_ret, ep_len, ep_newbytes, ep_bler = 0, 0, 0, 0, 0, 0, 0
                if epoch_ended:
                    sum_tx = info['total_txdata'].sum()
                    final_waiting = info['waitingdata'].sum()
                    ep_tx = epoch_tx / int(steps_per_epoch / max_ep_len)
                    ep_capacity = epoch_capacity / int(steps_per_epoch / max_ep_len)
                    ep_ret = epoch_reward / int(steps_per_epoch / max_ep_len)
                    ep_waiting = epoch_waiting / int(steps_per_epoch / max_ep_len)
                    ep_newbytes = epoch_newbytes / int(steps_per_epoch / max_ep_len)
                    ep_bler = epoch_bler / int(steps_per_epoch / max_ep_len)
                    ep_rbg_used = epoch_rbg_used / int(steps_per_epoch / max_ep_len)
                    if len(y) >= 2:
                        if y[-1] - y[-2] >= 15:
                            error = 11111
                    logger.store(Ep_ret=ep_ret, ep_bler=ep_r1, ep_fairness=ep_r2, Ep_tx=ep_tx, EP_new=ep_newbytes,
                                 EP_finalwiat=final_waiting,
                                 sumtx=sum_tx, Ep_capacity=ep_capacity, Ep_rbgused=ep_rbg_used, Ep_bler=ep_bler,
                                 Error=error)
                _, o = env.reset(ontime, offtime)
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)
        x.append(epoch + 1)
        y.append(ep_ret)
        ax.cla()  # clear plot
        ax.plot(x, y, 'r', lw=1)  # draw line chart
        plt.pause(0.1)
        plt.savefig('./test.jpg')
        start_time = time.time()
        update()
        end_time = time.time()
        # print("update time", end_time - start_time)
        logger.log_tabular('epoch       ', epoch)
        logger.log_tabular("ep_ret      ", ep_ret)
        logger.log_tabular("ep_bler      ", ep_r1)
        logger.log_tabular("ep_fair      ", ep_r2)
        logger.log_tabular("ep_tx       ", ep_tx)
        logger.log_tabular("newbytes    ", ep_newbytes)
        logger.log_tabular("final_wiating   ", final_waiting)
        logger.log_tabular('ep_rbg', ep_rbg_used)
        logger.log_tabular('ep_bler     ', ep_bler)
        logger.dump_tabular()


if __name__ == '__main__':

    from spinup.utils.run_utils import setup_logger_kwargs
    import os

    trace_dir = os.getcwd() + "/result"
    logger_kwargs = setup_logger_kwargs("ppo-ra", data_dir=trace_dir, datestamp=True)
    ppo(satellite_run,
        actor_critic=core.RA_ActorCritic, ac_kwargs={"hidden_sizes": (256, 512, 1024, 512, 256)},
        steps_per_epoch=200, epochs=500, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-4, train_pi_iters=50, train_v_iters=50, lam=0.97, max_ep_len=50,
        logger_kwargs=logger_kwargs, use_cuda=False)
