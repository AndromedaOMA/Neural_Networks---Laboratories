import flappy_bird_gymnasium
import gymnasium
import torch
import itertools
import random

from scripts import DuelingCNN, ReplayMemory, FrameStacker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class Agent2:
    def __init__(self, model_to_test):

        # Hyperparameters
        self.replay_memory_size = 200000
        self.mini_batch_size = 84
        self.epsilon_init = 1
        self.epsilon_decay = 0.998
        self.epsilon_min = 0.1
        self.network_sync_rate = 100
        self.learning_rate_a = 0.001
        self.discount_factor_g = 0.999
        self.stop_on_reward = 500
        self.best_reward = 5

        self.enable_double_dqn = True

        self.image_stack_dimension = 4
        self.model_to_test = model_to_test

        self.loss_fn = torch.nn.SmoothL1Loss()
        # self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None
        self.scheduler = None

        # criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        # optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    def run(self, is_training=True, render=False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)

        num_actions = env.action_space.n
        policy_dqn = DuelingCNN(input_channels=self.image_stack_dimension, input_size=64, out_layer_dim=2).to(device)

        if not is_training:
            policy_dqn.load_state_dict(torch.load(self.model_to_test))
            print("Loaded the trained Q-function for testing.")

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            # Duplicate the DQN for target_dqn where we are going to compute the Q values
            target_dqn = DuelingCNN(input_channels=self.image_stack_dimension, input_size=64, out_layer_dim=2).to(device)
            # Copy the w and b at target_dqn from policy_dqn
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)

        reward_per_episode = []
        epsilon_history = []
        frame_stacker = FrameStacker(stack_size=self.image_stack_dimension, height=64, width=64)

        for episode in itertools.count():
            state, _ = env.reset()
            stacked_frames = frame_stacker.reset(state)
            state = torch.tensor(stacked_frames, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated:
                # Epsilon-greedy action selection
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():  # ??
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax().item()
                        # action = policy_dqn(state.unsqueeze(dim=0)).argmax().item()

                # Environment step
                new_frame, reward, terminated, _, info = env.step(action)
                stacked_frames = frame_stacker.update(new_frame)
                episode_reward += reward

                new_state = torch.tensor(stacked_frames, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, torch.tensor(action), new_state, reward, terminated))
                    step_count += 1

                state = new_state

            reward_per_episode.append(episode_reward)

            # Update epsilon
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            # Train the network
            if is_training and len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Sync target network
                if step_count >= self.network_sync_rate:
                    # Copy the w and b at target_dqn from policy_dqn at each self.network_sync_rate
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                torch.save(policy_dqn.state_dict(), f"./best_models_CNN/DuelingCNN/trained_q_function_{self.best_reward:.3f}.pth")
                print(f"Model with best reward of {episode_reward} saved at episode {episode}.")

            print(f"Episode: {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")

            # Exit
            if is_training and episode_reward >= self.stop_on_reward:
                print(f"Training stopped after achieving reward threshold of {self.stop_on_reward}.")
                break

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        """
        Q-Learning Formula:
            q[state, action] = lr * (reward + discount_factor * max(q[new_state,:]) - q[state, action])
        ---------------------------------------------------------------------------------
        DQN Target Formula:
            Qt[state, action] = reward if new_state is terminal else
                               reward + discount_factor * max(Qt[new_state,:])
        ---------------------------------------------------------------------------------
        Double DQN Target Formula:
            best_action = arg(max(Qp[new_state,:]))
            Qt[state, action] = reward if new_state is terminal else
                               reward + discount_factor * max(Qt[best_action])
        """
        frames, actions, new_frames, rewards, terminations = zip(*mini_batch)

        frames = torch.stack(frames)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        new_frames = torch.stack(new_frames)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
            if self.enable_double_dqn:
                """Double DQN Target Formula"""
                best_action_from_policy = policy_dqn(new_frames).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_frames).gather(dim=1,
                                                                                                                 index=best_action_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                """DQN Target Formula"""
                target_q = rewards + (1 - terminations) * self.discount_factor_g * target_dqn(new_frames).max(dim=1)[0]

        # Compute Q values from policy
        current_q = policy_dqn(frames).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()
