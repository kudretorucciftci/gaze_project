import csv
import time

class RewardEngine:
    def __init__(self,
                 fixation_reward=1.0,
                 short_fix_penalty=-0.5,
                 noise_penalty=-1.0,
                 min_fix_duration=0.12):

        self.fixation_reward = fixation_reward
        self.short_fix_penalty = short_fix_penalty
        self.noise_penalty = noise_penalty
        self.min_fix_duration = min_fix_duration

        self.reward_file = "reward_log.csv"
        with open(self.reward_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "reward",
                "reason",
                "fix_duration"
            ])

    def evaluate(self, is_fixation, fixation_duration):
        reward = 0.0
        reason = "none"

        if is_fixation:
            # fixation devam ediyorsa henüz karar yok
            return 0.0

        if fixation_duration > 0:
            if fixation_duration >= self.min_fix_duration:
                reward = self.fixation_reward
                reason = "stable_fixation"
            else:
                reward = self.short_fix_penalty
                reason = "short_fixation"
        else:
            reward = self.noise_penalty
            reason = "noise_motion"

        self._log(reward, reason, fixation_duration)
        return reward

    def _log(self, reward, reason, duration):
        with open(self.reward_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                reward,
                reason,
                round(duration, 4)
            ])

