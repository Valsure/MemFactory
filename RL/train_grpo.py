from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
from reward import *
import os
import json
import swanlab


class GSM8KDataset(Dataset):
    
    def __init__(self, data_path, tokenizer):
        self.user_prompt = open("/aliyun-oss/ziheng/GSM8K_system_prompt.txt").read()
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f if line.strip()][4000:6000]
  
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        answer = sample['numeric_answer']
        prompt = self.user_prompt + '\n Here is the problem: \n' + sample['question']
        return {'prompt': prompt, 'answer': answer}



@dataclass
class Samples:
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any
    answer: Any
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    response_length: int


class GRPOArguments:
    import time
    output_dir = f'/home/deepspeed/workdir/output/RawGRPO/1.5B/{time.time()}'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 5e-7
    save_steps = 500
    epoch = 1
    num_generations = 8 
    max_prompt_length = 300
    max_generate_length = 600
    reward_weights : List[float] = None 
    beta = 0
    clip_eps = 0.2
    gradient_accumulation_steps = 1
    num_iterations = 1 # 采样一次样本训练模型轮数
    batch_size = 1

class GRPOTrainer:
    def __init__(self,
        model = None,
        reward_funcs: Union[List[str], List[Callable]] = None,
        args = None,
        train_dataset: Optional[Union[Dataset]] = None,
        eval_dataset: Optional[Union[Dataset]] = None,
        tokenizer = None,
        reward_tokenizers = None):

        self.args = args
        # 加载模型
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)
        
        # 是否使用参考模型
        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
    
        
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        self.tokenizer = self.get_tokenizer(tokenizer)
        
        
        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]
        
        
        self.reward_funcs = reward_funcs
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 缓存已经生成的数据的一个批次的数据，可供模型多次训练迭代，无需重新生成
        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        
        # 模型更新的次数
        self.update_steps = 0
        
        # 初始化 GradScaler (用于混合精度训练)
        self.scaler = torch.amp.GradScaler() if self.args.device == 'cuda' else None 
    def get_tokenizer(self, tokenizer):
        tokenizer.padding_side = "left"
        # 确保 tokenizer 有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    # 生成样本，以组为单位
    def generate_samples(self, batch_data):
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in batch_data['prompt']]
        answers = [None] * len(prompts)
        
        if 'answer' in batch_data:
            answers = [answer for answer in batch_data['answer']]
        
        max_length = self.args.max_generate_length + self.args.max_prompt_length
        
        for prompt, answer in zip(prompts, answers):
            messages = [{"role": "system", 'content': 'You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.'},
                        {"role": "user", 'content': prompt}]
            print(messages)
            input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            # 生成一个group的输入数据
            tokenized_inputs = self.tokenizer([input_text] * self.args.num_generations, padding='max_length', max_length=self.args.max_prompt_length, truncation=True, return_tensors='pt')
            prompt_ids = tokenized_inputs['input_ids']
            # import pdb; pdb.set_trace()
            with torch.no_grad():
                prompt_response_ids = self.model.generate(**tokenized_inputs.to(self.args.device), 
                                    max_new_tokens = self.args.max_generate_length,
                                    temperature=1)
                
            if prompt_response_ids.size(1) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                prompt_response_ids = torch.cat([prompt_response_ids, torch.full((prompt_response_ids.size(0), max_length - prompt_response_ids.size(1)), fill_value=self.tokenizer.pad_token_id, device=prompt_response_ids.device)], dim=1)
          
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            action_mask = (response_ids.ne(self.tokenizer.eos_token_id) & response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            

            # 存储的是一个group的数据
            samples = Samples(
                prompt_response_ids=prompt_response_ids,
                response_ids=response_ids,
                prompt = prompt,
                answer = answer,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                response_length=action_mask.float().sum(dim=-1)
            )
            samples_list.append(samples)

        return samples_list
    
    # 生成经验(优势、token的概率分布)
    def generate_experiences(self, batch_data):
        # TODO 主要是这里得改一下，rollout 就是正常的 rollout 就好，要加一个能够测试奖励的函数
        # TODO 在处理数据集的时候，要把原始数据集处理成支持 prompt 的形式，方便在 generate_samples 中使用
        # prompt--model--n*rollouts ，现在的问题是，rollouts 是抽取更新分开？还是一起？rollouts 应该长成什么样子，后续才能打分？
        self.model.eval()
        samples_list = self.generate_samples(batch_data)
        
        batch_prompt_response_ids = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_advantages = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []
        
        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids # shape: (num_generations, seq_len)
            response_ids = samples.response_ids # shape: (num_generations, seq_len)
            answer = samples.answer
            attention_mask = samples.attention_mask # shape: (num_generations, seq_len)
            action_mask = samples.action_mask # shape: (num_generations, seq_len)
            num_actions = samples.num_actions
            prompt = samples.prompt
            batch_prompt_response_ids.append(prompt_response_ids)
            batch_attention_mask.append(attention_mask)
            batch_action_mask.append(action_mask)
            
            with torch.no_grad():
                # 计算策略模型输出token的概率
                old_action_log_probs = self.get_action_log_probs(self.model, prompt_response_ids, attention_mask, num_actions)
                batch_old_action_log_probs.append(old_action_log_probs)
                
                # 是否使用参考模型
                if self.ref_model:
                    #计算参考模型输出token的概率
                    ref_action_log_probs = self.get_action_log_probs(self.ref_model, prompt_response_ids, attention_mask, num_actions)
                    batch_ref_action_log_probs.append(ref_action_log_probs)
                
                # 存储各个奖励函数在一个group内各个响应的奖励
                rewards_per_func = torch.zeros(len(self.reward_funcs), self.args.num_generations, device=self.args.device)
                
                # 将输出转换成文本
                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts = [prompt] * len(response_texts)
                prompt_response_texts = [prompt + response for prompt, response in zip(prompt_texts, response_texts)]
                
                for i, (reward_func) in enumerate(
                    self.reward_funcs
                ):
                    answers = [answer] * len(prompt_texts)
                    output_reward_func = reward_func(prompts=prompt_texts, responses=response_texts, answers=answers)
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[i] = torch.tensor(output_reward_func, dtype=torch.float32, device=self.args.device)
                
                for i, reward in enumerate(rewards_per_func):
                    swanlab.log({f"reward_{i}": reward.mean().item()})
                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                if len(self.args.reward_weights) != len(self.reward_funcs):
                    raise ValueError("The number of reward weights must be equal to the number of reward functions.")
                    
                rewards = rewards_per_func * torch.tensor(self.args.reward_weights, dtype=torch.float32, device=rewards_per_func.device).unsqueeze(1)

                # rewards: [num_funcs, num_generations]
                rewards = rewards.sum(dim=0) # shape: [num_generations]
                print(f'rewards: {rewards}')
                mean_group_rewards = rewards.mean()
                std_group_rewards = rewards.std()
                swanlab.log({"rewards_mean": mean_group_rewards.item(), "rewards_std": std_group_rewards.item()})

                # GRPO的优势是句子粒度的，而非token粒度的
                advantages = (rewards - mean_group_rewards) / (std_group_rewards + 1e-8) # shape: [num_generations]
                batch_advantages.append(advantages)
        
               
        return {
            "prompt_response_ids": torch.cat(batch_prompt_response_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "action_mask": torch.cat(batch_action_mask, dim=0),
            "old_action_log_probs": torch.cat(batch_old_action_log_probs, dim=0),
            "ref_action_log_probs": torch.cat(batch_ref_action_log_probs, dim=0) if self.ref_model else None,
            "advantages": torch.cat(batch_advantages, dim=0),
        }
        
    
    def compute_loss(self, model, inputs):
        
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
        
        k3 = None
        if self.args.beta != 0.0:
            if inputs.get('ref_action_log_probs') is None:
                raise ValueError("ref_action_log_probs is required when beta != 0.0, but ref_model is not set.")
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs 
            log_ratio = log_ratio * action_mask
            
            k3 = log_ratio.exp() - 1 - log_ratio
        
        advantages = inputs['advantages']
        
        old_action_log_probs = inputs['old_action_log_probs'] if self.args.num_iterations > 1 else action_log_probs.detach()
        coef_1 = torch.exp(action_log_probs - old_action_log_probs) # 重要性采样 shape: [batch_size * num_generations, num_actions]
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1) # 一个序列中每个token的优势是一样的
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask
        if self.args.beta != 0.0 and k3 is not None:
            per_token_loss = per_token_loss + self.args.beta * k3
        
        loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1) # shape: [batch_size * num_generations]
        loss = loss.mean()
        
        # loss = per_token_loss.sum() / action_mask.sum()
        
        return loss


    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        
        # 计算策略模型输出token的概率
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs

    
    
    def train_step(self, model, inputs, optimizer, step):
        model.train()
        
        # 记录未缩放的损失值
        raw_loss = None
        
        if self.scaler is not None:
            # 使用混合精度训练
            with torch.amp.autocast(device_type='cuda'):
                raw_loss = self.compute_loss(model, inputs)
                scaled_loss = raw_loss / self.args.gradient_accumulation_steps
                scaled_loss = self.scaler.scale(scaled_loss)
                scaled_loss.backward()
        else:
            # 不使用混合精度训练
            raw_loss = self.compute_loss(model, inputs)
            scaled_loss = raw_loss / self.args.gradient_accumulation_steps
            scaled_loss.backward()
        
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            # 梯度累积完成，更新模型
            if self.scaler is not None and self.args.device == 'cuda':
                self.scaler.unscale_(optimizer)
                # 可以在这里添加梯度裁剪
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
            
            # 记录未缩放的损失值
            loss_value = raw_loss.item() if raw_loss is not None else scaled_loss.item()
            swanlab.log({"grpo_loss": loss_value})
            print(f"step: {self.update_steps}/{self.global_steps}  grpo_loss: {loss_value:.8f}")
        
        if self.args.device == 'cuda':
            torch.cuda.empty_cache()

    def train(self):
        # 初始化优化器的梯度为零
        self.optimizer.zero_grad()
        
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        for epoch_idx in range(self.args.epoch):
            
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=False)
            for idx, batch in enumerate(dataloader):
                
                experiences = self.generate_experiences(batch)
                buffer_idx = idx % self.args.gradient_accumulation_steps
                self.input_buffer[buffer_idx] = experiences
                
                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                    # 梯度累积完成，开始训练
                    for iteration in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1
                        if self.update_steps % self.args.save_steps == 0:
                            os.makedirs(self.args.output_dir + f'/checkpoint_{self.update_steps}', exist_ok=True)
                            self.model.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                            self.tokenizer.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                        
                del experiences
    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)                 

if __name__ == "__main__":
    import os
    os.environ["SWANLAB_API_KEY"] = "Zkrggz0kWlnEuNRu5r4dz"
            
    args = GRPOArguments()
    
    swanlab.init(
    project="GRPO-1.5B-Raw",
    run_name="test",   
    config={
        "lr": args.lr,
        "batch_size": args.batch_size,
        "num_generations": args.num_generations
        }
    )
    tokenizer = AutoTokenizer.from_pretrained('/home/models/Qwen2.5-7B-Instruct/')
    model = AutoModelForCausalLM.from_pretrained('/home/models/Qwen2.5-7B-Instruct/')
    
    
    prompts_dataset = GSM8KDataset('/home/liziheng/datasets/train_with_numeric_answer.jsonl', tokenizer)
  
    trainer = GRPOTrainer(model=model,
                          reward_funcs = [correctness_reward, hard_format_reward],
                          args=args,
                          train_dataset=prompts_dataset,
                          tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()
    