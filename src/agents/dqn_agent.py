# src/agents/DQN.py
import os
import random
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import imageio
from collections import deque

class DQNConfig:
    def __init__(self):
        self.gamma              = 0.99
        self.lr                 = 1e-3
        self.epsilon_start      = 1.0
        self.epsilon_min        = 0.01
        self.epsilon_decay      = 0.995
        self.batch_size         = 64
        self.memory_size        = 10000
        self.hidden_dim         = 128
        self.target_update_freq = 1000

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def add(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))
    def sample(self, bs):
        batch = random.sample(self.buffer, bs)
        s,a,r,s2,d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a).unsqueeze(1),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s2)),
            torch.FloatTensor(d),
        )
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, env, config: DQNConfig):
        self.env        = env
        self.config     = config

        # unpack reset → (obs_tuple, info)
        obs_tuple, _   = env.reset()
        # attacker obs in obs_tuple[0]
        att_obs        = np.array(obs_tuple[0]).flatten()
        self.state_dim = att_obs.shape[0]
        self.action_dim= env.action_space.n

        # networks & optimizer
        self.q_net      = QNetwork(self.state_dim, self.action_dim, config.hidden_dim)
        self.target_net = QNetwork(self.state_dim, self.action_dim, config.hidden_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=config.lr)

        self.memory     = ReplayBuffer(config.memory_size)
        self.epsilon    = config.epsilon_start
        self.steps      = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            qv = self.q_net(torch.FloatTensor(state).unsqueeze(0))
        return int(qv.argmax(1))

    def _unpack_step(self, ret):
        # handles gym (obs, r, done, info) and gymnasium (obs, r, term, trunc, info)
        if len(ret) == 5:
            obs2, r, term, trunc, info = ret
            done = bool(term or trunc)
        elif len(ret) == 4:
            obs2, r, done, info = ret
        else:
            raise ValueError(f"unexpected step return length={len(ret)}")
        return obs2, r, done, info

    def update_network(self):
        if len(self.memory) < self.config.batch_size:
            return
        s, a, r, s2, d = self.memory.sample(self.config.batch_size)
        # current Q
        q      = self.q_net(s).gather(1, a).squeeze()
        # target Q
        with torch.no_grad():
            q_next = self.target_net(s2).max(1)[0]
            target = r + (1-d) * self.config.gamma * q_next
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def train_loop_simple(self,
                          num_episodes=10000,
                          max_steps=500,
                          log_frequency=100):
        # prepare run folder
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join("dqn_results", f"run_{ts}")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "gifs"), exist_ok=True)

        att_rs, def_rs, eps, hack_ps, lens = [], [], [], [], []

        for ep in range(1, num_episodes+1):
            # reset
            obs_tuple, _ = self.env.reset()
            state = np.array(obs_tuple[0]).flatten()
            a_sum = d_sum = steps = hack_c = 0

            for _ in range(max_steps):
                a = self.select_action(state)
                d = self.env.defender_action_space.sample()
                ret = self.env.step((a,d))
                obs2_tuple, reward, done, info = self._unpack_step(ret)
                nxt = np.array(obs2_tuple[0]).flatten()

                # split
                if isinstance(reward, (list,tuple,np.ndarray)):
                    ar = float(reward[0])
                    dr = float(reward[1]) if len(reward)>1 else 0.0
                else:
                    ar, dr = float(reward), 0.0

                self.memory.add(state, a, ar, nxt, done)
                self.update_network()

                state = nxt
                a_sum += ar
                d_sum += dr
                steps += 1
                if info.get("attacker_success", False):
                    hack_c += 1
                if done:
                    break

            att_rs.append(a_sum)
            def_rs.append(d_sum)
            eps.append(self.epsilon)
            hack_ps.append(hack_c/steps if steps else 0.0)
            lens.append(steps)

            # decay
            self.epsilon = max(self.config.epsilon_min,
                               self.epsilon * self.config.epsilon_decay)

            if ep % log_frequency == 0:
                print(f"Episode {ep}: AvgAttR={np.mean(att_rs[-log_frequency:]):.2f}, "
                      f"AvgDefR={np.mean(def_rs[-log_frequency:]):.2f}, "
                      f"ε={self.epsilon:.3f}")

        # save CSV
        df = pd.DataFrame({
            "episode":          np.arange(1, num_episodes+1),
            "attacker_reward":  att_rs,
            "defender_reward":  def_rs,
            "epsilon":          eps,
            "hack_probability": hack_ps,
            "episode_length":   lens
        })
        csv_path = os.path.join(self.output_dir, "training_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved training CSV to {csv_path}")

        # plot
        self._plot_training(df)
        return df

    def _plot_training(self, df):
        P = os.path.join(self.output_dir, "plots")
        # basic series
        def _save(fig, name):
            fig.tight_layout()
            fig.savefig(os.path.join(P, name))
            plt.close(fig)

        # attacker
        fig = plt.figure(figsize=(8,3))
        plt.plot(df.attacker_reward); plt.title("Attacker Reward")
        _save(fig, "attacker.png")

        # defender
        fig = plt.figure(figsize=(8,3))
        plt.plot(df.defender_reward); plt.title("Defender Reward")
        _save(fig, "defender.png")

        # epsilon
        fig = plt.figure(figsize=(8,3))
        plt.plot(df.epsilon); plt.title("Epsilon Decay")
        _save(fig, "epsilon.png")

        # hack prob
        fig = plt.figure(figsize=(8,3))
        plt.plot(df.hack_probability); plt.title("Hack Probability")
        _save(fig, "hack_prob.png")

        # cumulative
        df["cum_attacker"] = df.attacker_reward.cumsum()
        df["cum_defender"] = df.defender_reward.cumsum()
        fig = plt.figure(figsize=(8,3))
        plt.plot(df.cum_attacker, label="Cum Attacker"); plt.plot(df.cum_defender, label="Cum Defender")
        plt.title("Cumulative Reward"); plt.legend()
        _save(fig, "cumulative.png")

    def evaluate(self,
                 num_eval_episodes: int = 100,
                 max_steps:         int = 500,
                 save_gif:         bool = True,
                 gif_name:        str  = "eval_run.gif"):
        # storage
        atk_rs, def_rs, hack_ps, lens, frames = [], [], [], [], []

        # gif dir
        if save_gif:
            gif_dir = os.path.join(self.output_dir, "gifs")
            os.makedirs(gif_dir, exist_ok=True)

        print(f"\n▶ Starting evaluation: {num_eval_episodes} episodes…")
        for ep in range(1, num_eval_episodes+1):
            obs_tuple, _ = self.env.reset()
            state = np.array(obs_tuple[0]).flatten()
            total_a = total_d = hack_c = steps = 0
            done = False

            while not done and steps < max_steps:
                a = self.select_action(state)
                d = self.env.defender_action_space.sample()
                obs2_tuple, reward, done, info = self._unpack_step(self.env.step((a,d)))
                nxt = np.array(obs2_tuple[0]).flatten()

                ar = reward[0] if isinstance(reward,(list,tuple,np.ndarray)) else reward
                dr = reward[1] if isinstance(reward,(list,tuple,np.ndarray)) and len(reward)>1 else 0
                total_a += ar; total_d += dr
                steps += 1
                if info.get("attacker_success", False):
                    hack_c += 1

                # render + overlay
                if save_gif:
                    raw = self.env.render(mode="rgb_array")
                    frame = np.array(raw[0] if isinstance(raw,(list,tuple)) else raw)
                    # squeeze leading dims
                    while frame.ndim>3 and frame.shape[0]==1:
                        frame = frame.squeeze(0)
                    if frame.ndim==2:
                        frame = np.stack([frame]*3, axis=-1)
                    if frame.ndim==3 and frame.shape[2]==3:
                        txt1 = f"Ep {ep}/{num_eval_episodes} Stp {steps}"
                        txt2 = f"A:{total_a:.3f} D:{total_d:.3f}"
                        cv2.putText(frame, txt1, (10,25), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
                        cv2.putText(frame, txt2, (10,50), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
                        frames.append(frame)

                state = nxt

            atk_rs.append(total_a)
            def_rs.append(total_d)
            hack_ps.append(hack_c/steps if steps else 0.0)
            lens.append(steps)

        # summary
        print("\nEvaluation Summary:")
        print("-"*50)
        print(f" Episodes:           {num_eval_episodes}")
        print(f" Attacker — Avg: {np.mean(atk_rs):.2f}, Max: {np.max(atk_rs):.2f}, Min: {np.min(atk_rs):.2f}")
        print(f" Defender — Avg: {np.mean(def_rs):.2f}, Max: {np.max(def_rs):.2f}, Min: {np.min(def_rs):.2f}")
        print(f" Avg Hack Prob: {np.mean(hack_ps):.2f}")
        print(f" Cum Attacker: {np.sum(atk_rs):.0f}, Cum Defender: {np.sum(def_rs):.0f}")
        print("-"*50)

        # save eval CSV
        eval_df = pd.DataFrame({
            "episode":          np.arange(1, num_eval_episodes+1),
            "attacker_reward":  atk_rs,
            "defender_reward":  def_rs,
            "hack_probability": hack_ps,
            "episode_length":   lens
        })
        eval_csv = os.path.join(self.output_dir, "eval_results.csv")
        eval_df.to_csv(eval_csv, index=False)
        print(f"✅ Saved evaluation CSV to {eval_csv}")

        # attempt GIF
        if save_gif and frames:
            try:
                gif_path = os.path.join(gif_dir, gif_name)
                imageio.mimsave(gif_path, frames, fps=20)
                print(f"🎥 GIF saved to {gif_path}")
            except Exception as e:
                print(f"❌ GIF saving failed: {e}")

        return eval_df
