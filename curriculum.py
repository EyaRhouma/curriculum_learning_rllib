import gym

import ray
from ray import tune


class CarRacing(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make("CarRacing-v0")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.training_phase = 0
        self.steps = 10

    def set_phase(self, phase):
        if phase != self.training_phase:
            self.training_phase=phase
            if self.training_phase == 1:
                self.steps = 50
            elif self.training_phase == 2:
                self.steps = 100

            print ("Increasing training curriculum phase to: {}, steps: {} ".format(self.training_phase, self.steps))

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()


current_phase = 0

def on_train_result(info):
    global current_phase
    print("Manage Curriculum callback called on phase {}".format(current_phase))
    result = info["result"]
    if result["episode_reward_mean"] > 30:
        current_phase+=1
        print("info",info)
        trainer = info["trainer"]
        trainer.workers.foreach_worker(
                lambda ev: [e.set_phase(current_phase) for e in ev.async_env.get_unwrapped()])


if __name__ == "__main__":
    ray.init()
    tune.run_experiments({
        "test": {
            "run": "PPO",
            "env": CarRacing,
            "config": {
                "num_workers": 5,
                "callbacks": {
                    "on_train_result": tune.function(on_train_result),
                },
            },
        },
    })