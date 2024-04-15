
from src.rlhf.reward.reward_trainer import RewardTrainer
from src.rlhf.policy.policy_trainer import PolicyTrainer
from src.rlhf.reward_model import RewardModel
import torch
import copy

def ppo_train_reward(opt, model, tokenizer, accelerator):
    reward_model = RewardModel(opt, model).cuda()
    trainer = RewardTrainer(opt, reward_model, tokenizer, accelerator)
    trainer.train()
    print('finished RewardTrainer train.')


def ppo_train_policy(opt, model, tokenizer, accelerator):
    reward_model = RewardModel(opt, model).cuda()
    # rw_static = torch.load(f'{opt.model_path}/reward_model.pt')
    # reward_model.load_state_dict(rw_static, strict=False)

    policy_model = model
    ref_model = copy.deepcopy(model)
    critic_model = copy.deepcopy(reward_model)

    trainer = PolicyTrainer(opt, 
                            policy_model, 
                            ref_model, 
                            critic_model, 
                            reward_model, 
                            tokenizer,
                            accelerator)

    trainer.train()
    print('finished RLHFTrainer train.')