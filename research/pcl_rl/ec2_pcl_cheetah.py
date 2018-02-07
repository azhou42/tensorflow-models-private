import sys
import os

from rllab.misc.instrument import VariantGenerator, variant, run_experiment_lite

from pcl_rl.train_pcl import run

class VG(VariantGenerator):
    @variant
    def max_divergence(self):
        return [0.001, 0.0005, 0.002]

    @variant
    def rollout(self):
        return [1, 5, 10]

    @variant
    def seed(self):
        return [1, 11, 21]

def main():
    variants = VG().variants()
    for v in variants:
        variant = dict(
            batch_size=1,
            num_steps=10000000,
            env_str="HalfCheetah-v1",
            max_step=10,
            max_divergence=v['max_divergence'],
            rollout=v['rollout'],
            seed=v['seed']
        )
        run_experiment_lite(
            run,
            exp_prefix="trust_pcl_sweep_hi3",
            n_parallel=1,
            snapshot_mode="last",
            seed=v['seed'],
            mode='ec2',
            variant=variant,
            use_cloudpickle=True,
        )
        sys.exit()

if __name__ == "__main__":
    main()
