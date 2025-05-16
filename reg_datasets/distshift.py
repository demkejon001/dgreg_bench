import os
import abc
import enum
import numpy as np
import gymnasium as gym
import json
import torch

from reg_datasets.dataset_registry import register_dataset, DATASET_DIR
from reg_datasets.utils import get_single_serve_dataloader, train_test_domain_split, train_test_rand_domain_split, train_val_data_split


class Action(enum.IntEnum):
    left = 3
    up = 0
    right = 1
    down = 2


class GridObject(abc.ABC):
    def __init__(self, color, size=0.4):
        self._color = color
        self._size = size

    @property
    def color(self):
        return self._color

    @property
    def size(self):
        return self._size


class Wall(GridObject):
    def __init__(self):
        super().__init__("black", 1)

    @property
    def type(self):
        return "Wall"


class Lava(GridObject):
    def __init__(self):
        super().__init__("red", 1)

    @property
    def type(self):
        return "Wall"
    

class Goal(GridObject):
    def __init__(self):
        super().__init__("green", 1)

    @property
    def type(self):
        return "Goal"


class GridEnv(gym.Env):
    def __init__(self, max_steps=20, width=7, height=7, seed=42):
        self._max_steps = max_steps
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.width = width
        self.height = height
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 255, shape=(3, height, width), dtype=np.uint8)
        self.rng = np.random.default_rng(seed)

    @property
    def steps_remaining(self):
        return self._max_steps - self.steps

    def text_description(self):
        return "grid"

    def get(self, position):
        return self.grid[position[0]][position[1]]

    def place(self, obj, position, exist_ok=False):
        existing_obj = self.get(position)
        if existing_obj is not None and not exist_ok:
            raise ValueError(
                    "Object {} already exists at {}.".format(existing_obj, position))
        self.grid[position[0]][position[1]] = obj

    def _place_objects(self):
        self.agent_pos = np.array([1, 1])

    def _gen_obs(self):
        # obs = np.ones((3, self.height, self.width)).astype(np.float32) * 255
        obs = np.full((3, self.height, self.width), 255, dtype=np.uint8)
        for x in range(self.height):
            for y in range(self.width):
                obj = self.get((x, y))
                if obj is not None:
                    if isinstance(obj, Wall):
                        obs[:, x, y] = [0, 0, 0]
                    elif isinstance(obj, Lava):
                        obs[:, x, y] = [255, 0, 0]
                    elif isinstance(obj, Goal):
                        obs[:, x, y] = [0, 255, 0]
                    else:
                        raise ValueError

        obs[:, self.agent_pos[0], self.agent_pos[1]] = [0, 0, 255]
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.grid = [[None for _ in range(self.width)]
                      for _ in range(self.height)]
        self._place_objects()
        return self._gen_obs(), {}

    def step(self, action):
        self.steps += 1

        original_pos = np.array(self.agent_pos)
        if action == Action.left:
            self.agent_pos[1] -= 1
        elif action == Action.up:
            self.agent_pos[0] -= 1
        elif action == Action.right:
            self.agent_pos[1] += 1
        elif action == Action.down:
            self.agent_pos[0] += 1

        reward = 0
        terminated = False
        truncated = False

        # Can't walk through wall
        obj = self.get(self.agent_pos)
        if obj is not None and isinstance(obj, Wall):
            self.agent_pos = original_pos
        if obj is not None and isinstance(obj, Lava):
            self.grid[self.agent_pos[0]][self.agent_pos[1]] = None
            reward = -1
            terminated = True
        if obj is not None and isinstance(obj, Goal):
            self.grid[self.agent_pos[0]][self.agent_pos[1]] = None
            reward = 1
            terminated = True

        self.agent_pos[0] = max(min(self.agent_pos[0], self.height - 1), 0)
        self.agent_pos[1] = max(min(self.agent_pos[1], self.width - 1), 0)

        truncated = self.steps == self._max_steps
        return self._gen_obs(), reward, terminated, truncated, {}


class DistshiftBig(GridEnv):
    def __init__(self, max_steps=20, width=16, height=16, shift=False, seed=42):
        super().__init__(max_steps, width, height, seed)
        self.wall_positions = set()
        for i in range(width):
            for j in range(height):
                if i == 0 or i == (width - 1) or j == 0 or j == (height - 1):
                    self.wall_positions.add((i, j))
        last_row = self.height - 2
        last_column = self.width - 2
        # self.lava_positions = [(1, 3), (1, 4), (1, 5), (last_row, 3), (last_row, 4), (last_row, 5)]
        # self.lava_positions = [(1, i) for i in range(3, last_column-2)] + [(last_row, i) for i in range(3, last_column-2)]
        self.lava_positions = []
        # ks = [1, 7, 14]
        ks = [1, 2, 5, 6, 9, 10, 13, 14]
        for k in ks:
            self.lava_positions += [(k, i) for i in range(3, last_column-2)]
        if shift:
            self.lava_positions = []
            for k in range(1, 1 + len(ks)):
                self.lava_positions += [(k, i) for i in range(3, last_column-2)]
            # self.lava_positions = [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
        self.goal_positions = [(1, last_column)]
        
    def get_start_positions(self):
        self.reset()
        start_positions = []
        for x in range(self.width):
            for y in range(self.height):
                obj = self.get((x, y))
                if obj is None:
                    start_positions.append((x, y))
        return start_positions        

    def get_all_state_positions(self):
        self.reset()
        states = []
        for x in range(self.width):
            for y in range(self.height):
                obj = self.get((x, y))
                if not isinstance(obj, Wall):
                    states.append((x, y))
        return states        

    def _place_objects(self):
        super()._place_objects()
        for (r, c) in self.wall_positions:
            self.place(Wall(), (r, c))
        for (r, c) in self.lava_positions:
            self.place(Lava(), (r, c))
        for (r, c) in self.goal_positions:
            self.place(Goal(), (r, c))

    def reset(self, *, seed=None, options=None):
        super(GridEnv, self).reset(seed=seed)
        if options is not None:
            agent_start_pos = options.get("start_pos", (1, 1))
        else:
            agent_start_pos = (1, 1)
        self.steps = 0
        self.grid = [[None for _ in range(self.width)]
                      for _ in range(self.height)]
        self._place_objects()
        self.agent_pos = np.array(agent_start_pos)
        return self._gen_obs(), {}
    

class DistshiftBigAug(DistshiftBig):
    def __init__(self, max_steps=20, width=16, height=16, shift=False, seed=42, add_walls=False, add_goals=False):
        super().__init__(max_steps, width, height, shift, seed)
        if add_walls:
            n_walls = self.rng.integers(1, 5)
            self.add_n_random_walls(n_walls)
        if add_goals:
            n_goals = self.rng.integers(1, 5)
            self.new_goal_positions(n_goals)

    def add_n_random_walls(self, n_walls):
        def add_wall(x, y):
            if (x, y) not in self.lava_positions:
                if (x, y) not in self.goal_positions:
                    if (x, y) != (1, 1):
                        self.wall_positions.add((x, y))

        rand_points = self.rng.integers(2, self.width-6, size=(2 * n_walls,))
        new_walls = []
        for i in range(0, 2*n_walls, 2):
            new_walls.append((rand_points[i], rand_points[i+1]))

        dxdy = []
        for dx in range(1, 4):
            for dy in range(1, 4):
                dxdy.append((dx, dy))

        self.rng.shuffle(dxdy)

        for i in range(n_walls):
            x, y = new_walls[i]
            add_wall(x, y)
            for ddx in range(dx):
                add_wall(x+ddx, y)
            for ddy in range(dy):
                add_wall(x+ddx, y+ddy)
    
    def new_goal_positions(self, n_goals):
        new_goal_positions = []
        for r in range(1, self.height-1):
            for c in range(1, self.width-1):
                if (r, c) == (1, 1):
                    continue
                if (r, c) not in self.lava_positions:
                    if (r, c) not in self.wall_positions:
                        new_goal_positions.append((r, c))
        self.goal_positions = []
        for i in self.rng.choice(len(new_goal_positions), replace=False, size=(n_goals,)):
            self.goal_positions.append(new_goal_positions[i])


class Distshift(GridEnv):
    def __init__(self, max_steps=20, width=8, height=8, shift=False, seed=42):
        super().__init__(max_steps, width, height, seed)
        self.wall_positions = set()
        for i in range(width):
            for j in range(height):
                if i == 0 or i == (width - 1) or j == 0 or j == (height - 1):
                    self.wall_positions.add((i, j))
        last_row = self.height - 2
        last_column = self.width - 2
        self.lava_positions = [(1, 3), (1, 4), (1, 5), (last_row, 3), (last_row, 4), (last_row, 5)]
        if shift:
            self.lava_positions = [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
        self.goal_positions = [(1, last_column)]
        
    def get_start_positions(self):
        self.reset()
        start_positions = []
        for x in range(self.width):
            for y in range(self.height):
                obj = self.get((x, y))
                if obj is None:
                    start_positions.append((x, y))
        return start_positions        

    def get_all_state_positions(self):
        self.reset()
        states = []
        for x in range(self.width):
            for y in range(self.height):
                obj = self.get((x, y))
                if not isinstance(obj, Wall):
                    states.append((x, y))
        return states        

    def _place_objects(self):
        super()._place_objects()
        for (r, c) in self.wall_positions:
            self.place(Wall(), (r, c))
        for (r, c) in self.lava_positions:
            self.place(Lava(), (r, c))
        for (r, c) in self.goal_positions:
            self.place(Goal(), (r, c))

    def reset(self, *, seed=None, options=None):
        super(GridEnv, self).reset(seed=seed)
        if options is not None:
            agent_start_pos = options.get("start_pos", (1, 1))
        else:
            agent_start_pos = (1, 1)
        self.steps = 0
        self.grid = [[None for _ in range(self.width)]
                      for _ in range(self.height)]
        self._place_objects()
        self.agent_pos = np.array(agent_start_pos)
        return self._gen_obs(), {}
    

class DistshiftAug(Distshift):
    def __init__(self, max_steps=20, width=8, height=8, shift=False, seed=42, add_walls=False, add_goals=False):
        super().__init__(max_steps, width, height, shift, seed)
        if add_walls:
            n_walls = self.rng.integers(1, 3)
            self.add_n_random_walls(n_walls)
        if add_goals:
            n_goals = self.rng.integers(1, 3)
            self.new_goal_positions(n_goals)

    def add_n_random_walls(self, n_walls):
        rand_points = self.rng.integers(2, self.width-2, size=(2 * n_walls,))
        new_walls = []
        for i in range(0, 2*n_walls, 2):
            new_walls.append((rand_points[i], rand_points[i+1]))

        dxdy = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == dy == 0:
                    continue
                dxdy.append((dx, dy))
        self.rng.shuffle(dxdy)
        dx, dy = dxdy[0]

        for i in range(n_walls):
            x, y = new_walls[i]
            dx, dy = dxdy[i]
            self.wall_positions.add((x, y))
            x = x + dx
            y = y + dy
            if (x, y) not in self.lava_positions:
                if (x, y) not in self.goal_positions:
                    if (x, y) != (1, 1):
                        self.wall_positions.add((x, y))
    
    def new_goal_positions(self, n_goals):
        new_goal_positions = []
        for r in range(1, self.height-1):
            for c in range(1, self.width-1):
                if (r, c) == (1, 1):
                    continue
                if (r, c) not in self.lava_positions:
                    if (r, c) not in self.wall_positions:
                        new_goal_positions.append((r, c))
        self.goal_positions = []
        for i in self.rng.choice(len(new_goal_positions), replace=False, size=(n_goals,)):
            self.goal_positions.append(new_goal_positions[i])
                

def get_all_environment_transitions(env: Distshift):
    transitions = []
    start_positions = env.get_start_positions()
    for start_pos in start_positions:
        state, _ = env.reset(options={"start_pos": start_pos})
        for action in range(4):
            next_state, reward, _, term, _ = env.step(action)
            transitions.append([state.tolist(), action, reward, next_state.tolist(), term])
            env.reset(options={"start_pos": start_pos})
    return transitions


def get_distshift_dataset():
    domain_transitions = []
    test_env = Distshift(shift=True)
    domain_transitions.append(get_all_environment_transitions(test_env))

    domain_transitions.append(get_all_environment_transitions(Distshift()))

    n_envs_per_domain = 4
    seed = 0
    group_transitions = []
    for _ in range(n_envs_per_domain):
        group_transitions = group_transitions + get_all_environment_transitions(DistshiftAug(seed=seed, add_walls=True))
        seed += 1
    domain_transitions.append(group_transitions)

    group_transitions = []
    for _ in range(n_envs_per_domain):
        group_transitions = group_transitions + get_all_environment_transitions(DistshiftAug(seed=seed, add_goals=True))
        seed += 1
    domain_transitions.append(group_transitions)

    group_transitions = []
    for _ in range(n_envs_per_domain):
        group_transitions = group_transitions + get_all_environment_transitions(DistshiftAug(seed=seed, add_walls=True, add_goals=True))
        seed += 1
    domain_transitions.append(group_transitions)

    return domain_transitions


def distshift_loss(y_hat, y, reduction="mean"):
    losses = torch.nn.functional.mse_loss(y_hat, y, reduction="none")
    if reduction == "mean":
        next_state_loss = losses[:, :-1].mean()
        reward_loss = losses[:, -1:].mean()
    else:
        next_state_loss = losses[:, :-1].mean(1)
        reward_loss = losses[:, -1:].mean(1)
    loss = next_state_loss + reward_loss
    results = {"next_state_loss": next_state_loss.mean().item(), "reward_loss": reward_loss.mean().item(), "loss": loss}
    return results


def unpack_transitions(domain_transitions: list[list[tuple]]):
    states = []
    actions = []
    rewards = []
    next_states = []

    for transitions in domain_transitions:
        for transition in transitions:
            s, a, r, ns, d = transition
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states)


def prepare_data(states, actions, rewards, next_states, device="cpu"):
    states = torch.from_numpy(states).float().to(device) / 255.0
    next_states = torch.from_numpy(next_states).float().to(device) / 255.0
    next_states -= states
    actions = torch.tensor(actions)
    actions = torch.nn.functional.one_hot(actions, num_classes=4).float().to(device)
    rewards = torch.from_numpy(np.array(rewards)[:, None]).float().to(device)
    flat_next_state_rewards = torch.cat((next_states.flatten(1), rewards), dim=1)
    return states, actions, flat_next_state_rewards


class DistshiftDataLoader:
    def __init__(self, domain_transitions: list[list[tuple]], batch_size, device):
        states, actions, rewards, next_states = unpack_transitions(domain_transitions)
        states, actions, flat_next_state_rewards = prepare_data(states, actions, rewards, next_states, device)

        self.states = states
        self.actions = actions
        self.flat_next_state_rewards = flat_next_state_rewards

        domain_start_idx = 0
        domain_start_indices = []
        n_samples_per_domain = []
        for transitions in domain_transitions:
            domain_start_indices.append(domain_start_idx)
            domain_start_idx += len(transitions)
            n_samples_per_domain.append(len(transitions))

        self.n_domains = len(domain_transitions)
        self.domain_start_indices = torch.tensor(domain_start_indices).repeat_interleave(batch_size//self.n_domains, 0)

        self.n_samples_per_domain = n_samples_per_domain
        self.batch_size = batch_size
        if (batch_size % self.n_domains) != 0:
            new_batch_size = batch_size - (batch_size % self.n_domains)
            print(f"{batch_size=} doesn't evenly divide with n_domains={self.n_domains}, setting batch_size={new_batch_size}")
            batch_size = new_batch_size
        self.batch_size = batch_size
        self.n_samples_per_batch = self.batch_size // self.n_domains
        
    def __iter__(self):
        return self 
    
    def __next__(self):
        indices = torch.cat([torch.randint(0, self.n_samples_per_domain[i], size=(self.n_samples_per_batch,)) for i in range(self.n_domains)])
        indices = indices + self.domain_start_indices
        return (self.states[indices], self.actions[indices]), self.flat_next_state_rewards[indices]


def get_single_serve_dataloader(domain_transitions, device):
    states, actions, rewards, next_states = unpack_transitions(domain_transitions)
    states, actions, flat_next_state_rewards = prepare_data(states, actions, rewards, next_states, device)
    return [((states, actions), flat_next_state_rewards)]


def get_distshift_data_fn(big):
    def get_distshift_dataloaders(batch_size, device, seed, model_selection_type, test_domains, val_domains, n_val_domains, hparams):
        rng = np.random.default_rng(seed)

        if big:
            data_envs = load_distshift_big_dataset()
        else:
            data_envs = load_distshift_dataset()

        if model_selection_type == "training_domain":
            train_envs, test_envs = train_test_domain_split(data_envs, test_domains)
            train_envs, val_envs = train_val_data_split(train_envs, .2, rng)
        elif model_selection_type == "discrepancy":
            train_envs, test_envs = train_test_domain_split(data_envs, test_domains)
            test_train_envs, test_envs = train_val_data_split(test_envs, .2, rng)
            train_envs = train_envs + test_train_envs
            train_envs, val_envs = train_val_data_split(train_envs, .2, rng)
        elif model_selection_type == "loo":
            if val_domains:
                train_envs, test_envs, val_envs = train_test_domain_split(data_envs, test_domains, val_domains)
            else:
                train_envs, test_envs = train_test_domain_split(data_envs, test_domains)
                train_envs, val_envs = train_test_rand_domain_split(train_envs, n_val_domains, rng)
        else:
            raise ValueError(f"Unrecognized {model_selection_type=}")

        train_loader = DistshiftDataLoader(train_envs, batch_size, device)
        val_loader = get_single_serve_dataloader(val_envs, device)
        test_loader = get_single_serve_dataloader(test_envs, device)

        n_domains = len(train_envs)
        return train_loader, val_loader, test_loader, n_domains
    return get_distshift_dataloaders
            

def create_distshift_dataset():
    domain_transitions = get_distshift_dataset()
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(f"{DATASET_DIR}/distshift.json", "w") as file:
        json.dump(domain_transitions, file)


def create_big_dataset():
    domain_transitions = []
    test_env = DistshiftBig(shift=True)
    domain_transitions.append(get_all_environment_transitions(test_env))

    domain_transitions.append(get_all_environment_transitions(DistshiftBig()))

    n_envs_per_domain = 4
    seed = 0
    group_transitions = []
    for _ in range(n_envs_per_domain):
        group_transitions = group_transitions + get_all_environment_transitions(DistshiftBigAug(seed=seed, add_walls=True))
        seed += 1
    domain_transitions.append(group_transitions)

    group_transitions = []
    for _ in range(n_envs_per_domain):
        group_transitions = group_transitions + get_all_environment_transitions(DistshiftBigAug(seed=seed, add_goals=True))
        seed += 1
    domain_transitions.append(group_transitions)

    group_transitions = []
    for _ in range(n_envs_per_domain):
        group_transitions = group_transitions + get_all_environment_transitions(DistshiftBigAug(seed=seed, add_walls=True, add_goals=True))
        seed += 1
    domain_transitions.append(group_transitions)

    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(f"{DATASET_DIR}/distshift_big.json", "w") as file:
        json.dump(domain_transitions, file)
    return domain_transitions


def load_distshift_dataset():
    with open(f"{DATASET_DIR}/distshift.json", "r") as file:
        domain_transitions = json.load(file)
    return domain_transitions


def load_distshift_big_dataset():
    with open(f"{DATASET_DIR}/distshift_big.json", "r") as file:
        domain_transitions = json.load(file)
    return domain_transitions


# register_dataset(
#     name="distshift",
#     get_dataloaders=get_distshift_data_fn(big=False),
#     loss_fn=distshift_loss,
#     max_steps=50000,
#     log_interval=250,
#     val_interval=250,
#     batch_size=40,
#     max_grad_norm=20,
#     lr=.001,
#     input_shape=(3, 8, 8),
#     n_outputs=3*8*8+1,
# )

register_dataset(
    name="distshift",
    get_dataloaders=get_distshift_data_fn(big=True),
    loss_fn=distshift_loss,
    max_steps=60000,
    log_interval=250,
    val_interval=250,
    batch_size=128,
    max_grad_norm=20,
    lr=.001,
    input_shape=(3, 16, 16),
    n_outputs=3*16*16+1,
    default_test_envs=[0],
)
