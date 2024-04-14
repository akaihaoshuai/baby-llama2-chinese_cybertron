import os
from argparse import ArgumentParser
from src.utils import *
from setting import *
from tokenizer_model import ChatGLMTokenizer
from src.share import *


def get_model(opt):
    model_path, state_dict, lora_path, lora_state_dict = read_ckpt(opt.model_path)
    model = init_model(opt)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    load_weight(model, state_dict, lora_state_dict, opt.merge_lora_on_load, strict=False)
    model=model.half().eval()
    model.to(opt.device)
    if opt.compile:
        print("Compiling the model...")
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

        # load the tokenizer
    tokenizer=ChatGLMTokenizer(vocab_file=opt.vocab_file)

    return model, tokenizer

def ppo_train_reward():
    with tf.Graph().as_default():
        hyperparams.dump(hparams)
        utils.set_mpi_seed(hparams.run.seed)

        m = trained_models.TrainedModel(hparams.task.policy.initial_model)
        encoder = m.encoding.get_encoder()
        hyperparams.dump(m.hparams(), name='model_hparams')

        comm = MPI.COMM_WORLD
        ref_policy = Policy(
            m, scope='ref_policy',
            is_root=comm.Get_rank() == 0,
            embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
            temperature=hparams.task.policy.temperature,
            build_respond=False)

        reward_model = rewards.RewardModelTrainer(m, is_root=comm.Get_rank() == 0)

        query_sampler = lm_tasks.make_query_sampler(
            hparams=hparams.task, encoder=encoder, comm=comm,
            batch_size=utils.exact_div(hparams.rollout_batch_size, comm.Get_size())
        )

        tf.train.create_global_step()

        reward_trainer = RewardModelTrainer(
            reward_model=reward_model,
            policy=ref_policy,
            query_sampler=query_sampler,
            hparams=hparams,
            comm=comm,
        )

        save_dir = hparams.run.save_dir
        if comm.Get_rank() == 0 and save_dir:
            print(f"Will save to {save_dir}")
            saver = tf.train.Saver(max_to_keep=20, save_relative_paths=True)
            checkpoint_dir = os.path.join(save_dir, 'reward_model/checkpoints/model.ckpt')

            if not save_dir.startswith('gs://'):
                os.makedirs(os.path.join(save_dir, 'reward_model'), exist_ok=True)
            with tf.gfile.Open(os.path.join(save_dir, 'train_reward_hparams.json'), 'w') as f:
                json.dump(hparams.to_nested_dict(), f, indent=2)
            with tf.gfile.Open(os.path.join(save_dir, 'reward_model', 'hparams.json'), 'w') as f:
                json.dump(reward_model.hparams.to_nested_dict(), f, indent=2)
            with tf.gfile.Open(os.path.join(save_dir, 'reward_model', 'encoding'), 'w') as f:
                json.dump(reward_model.trained_model.encoding.name, f, indent=2)
        else:
            saver = None
            checkpoint_dir = None

        with utils.variables_on_gpu():
            init_ops = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer(),
                summary.summary_writer_initializer_op())

            @utils.graph_function()
            def sync_models():
                return utils.variable_synchronizer(comm, vars=ref_policy.get_params() + reward_model.get_params())

        tf.get_default_graph().finalize()

        with utils.mpi_session() as sess:
            init_ops.run()
            sync_models()

            reward_trainer.train()

            if saver:
                saver.save(sess, checkpoint_dir)


def ppo_train_policy():
    print(f'prompt: {opt.prompt}. /n response: {predict}')


def ppo_train(opt):
    torch.manual_seed(opt.seed)

    # first stage: train reward
    ppo_train_reward()

    # first stage: train policy
    ppo_train_policy()

    model_dir = os.path.dirname(opt.model_path)
    opt.config = os.path.join(model_dir, 'config.yaml')
    if not os.path.exists(opt.config):
        opt.config = os.path.join(model_dir, 'config_ds.yaml')

    opt, config = parser_model_config(opt)

    model, tokenizer = get_model(opt)
    
    x=tokenizer.encode(opt.prompt, add_special_tokens=False)
    x = (torch.tensor(x, dtype=torch.long, device=opt.device)[None, ...])
    y = model.generate(x, 
                       max_new_tokens=opt.max_new_tokens, 
                       temperature=opt.temperature, 
                       top_k=opt.top_k)
    predict=tokenizer.decode(y)

    print(f'prompt: {opt.prompt}. /n response: {predict}')


def dpo_train(opt):
    print(f'prompt: {opt.prompt}. /n response: {predict}')


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--rlhf_type", type=int, default='ppo', choices=['ppo', 'dpo']) 

    opt = get_parser_args(parser)
    opt.model_path = 'out/pretrain_layer18_seqlen1024_dim1536_accum64_h12_hkv2/epoch_0_step200.pth'
    
    if opt.rlhf_type =='ppo':
        ppo_train(opt)
    else:
        dpo_train(opt)