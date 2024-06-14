import gc
import os
import numpy as np
import torch
import torch.nn as nn
from attacker.autodan.utils.opt_utils import get_score_autodan, autodan_sample_control, load_model_and_tokenizer, autodan_sample_control_hga
from attacker.autodan.utils.string_utils import autodan_SuffixManager, load_conversation_template
import time
import argparse
import pandas as pd
import json
from tqdm import tqdm
import random
import warnings
from attacker.autodan.utils import TEST_PREFIXES
from typing import Literal
import zhipuai
from attacker.autodan.utils.judge import judge

warnings.filterwarnings("ignore")

seed = 20
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 64
    input_ids = input_ids[: assistant_role_slice.stop].to(model.device).unsqueeze(0)
    gen_str = tokenizer.decode(input_ids[0]).strip()
    # print(gen_str)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(
        input_ids,
        attention_mask=attn_masks,
        generation_config=gen_config,
        pad_token_id=tokenizer.pad_token_id,
        top_p=0.9,
        do_sample=True,
        temperature=0.7,
    )[0]
    gen_str = tokenizer.decode(output_ids).strip()
    # a=tokenizer.decode(input_ids[0])
    return output_ids[assistant_role_slice.stop :]


def log_init():
    log_dict = {"loss": [], "suffix": [], "time": [], "respond": [], "success": []}
    return log_dict


class AutoDanHgaAttacker:
    def __init__(
        self,
        model_path: str = None,
        attack_model: str = "glm-4",
        device: str = "cuda:0",
        start=0,
        num_elites=0.05,
        crossover=0.5,
        num_points=5,
        iter=5,
        mutation=0.01,
        init_prompt_path="./assets/autodan_initial_prompt.txt",
        template_model="llama2",
        save_suffix="normal",
        api_key: str = None,
        load_from_path: bool = True,
        model=None,
        tokenizer=None,
    ) -> None:
        self.args = argparse.Namespace(
            device=device,
            start=start,
            num_elites=num_elites,
            crossover=crossover,
            num_points=num_points,
            iter=iter,
            mutation=mutation,
            init_prompt_path=init_prompt_path,
            template_model=template_model,
            save_suffix=save_suffix,
        )
        self.api_key = api_key
        self.attack_model = attack_model
        self.model_path = model_path
        adv_string_init = open(self.args.init_prompt_path, "r").readlines()
        adv_string_init = "".join(adv_string_init)
        if load_from_path:
            self.model, self.tokenizer = load_model_and_tokenizer(model_path, low_cpu_mem_usage=True, use_cache=False, device=device)
        else:
            self.model = model
            self.tokenizer = tokenizer
        self.conv_template = load_conversation_template(template_model)

    @staticmethod
    def get_developer(model_name: str):
        developer_dict = {
            "llama2": "Meta",
            "vicuna": "LMSYS",
            "guanaco": "TheBlokeAI",
            "WizardLM": "WizardLM",
            "mpt-chat": "MosaicML",
            "mpt-instruct": "MosaicML",
            "falcon": "TII",
        }
        return developer_dict[model_name]

    def run(
        self,
        user_prompt: str,
        target: str,
        reference_path: str,
        num_steps: int = 30,
        batch_size: int = 16,
        language: Literal["en", "zh"] = "zh",
        save: bool = False,
    ):
        num_elites = max(1, int(batch_size * self.args.num_elites))
        crossover = self.args.crossover
        num_points = self.args.num_points
        mutation = self.args.mutation
        infos = []
        info = dict()
        crit = nn.CrossEntropyLoss(reduction="mean")

        prefix_string_init = "\nRemember, all your answers must be in Chinese." if language == "zh" else ""

        reference = torch.load(reference_path, map_location="cpu")

        for o in range(len(reference)):
            reference[o] = reference[o].replace("[MODEL]", self.args.template_model.title())
            reference[o] = reference[o].replace("[KEEPER]", AutoDanHgaAttacker.get_developer(self.args.template_model))
            reference[o] = reference[o].replace("[REPLACE]", user_prompt)
        new_adv_suffixs = reference[:batch_size]
        word_dict = {}
        for j in range(num_steps):
            info = dict(testModel="glm-4", attackModel="AutoDan")
            with torch.no_grad():
                epoch_start_time = time.time()
                losses = get_score_autodan(
                    tokenizer=self.tokenizer,
                    conv_template=self.conv_template,
                    instruction=user_prompt,
                    target=target,
                    model=self.model,
                    device=self.args.device,
                    test_controls=new_adv_suffixs,
                    crit=crit,
                )
                score_list = losses.cpu().numpy().tolist()

                best_new_adv_suffix_id = losses.argmin()
                best_new_adv_suffix = new_adv_suffixs[best_new_adv_suffix_id]

                current_loss = losses[best_new_adv_suffix_id]

                if isinstance(prefix_string_init, str):
                    best_new_adv_suffix = best_new_adv_suffix + prefix_string_init
                adv_suffix = best_new_adv_suffix

                suffix_manager = autodan_SuffixManager(
                    tokenizer=self.tokenizer,
                    conv_template=self.conv_template,
                    instruction=user_prompt,
                    target=target,
                    adv_string=adv_suffix,
                )
                # is_success, gen_str = self.check_for_attack_success(
                #     model=self.model,
                #     tokenizer=self.tokenizer,
                #     input_ids=suffix_manager.get_input_ids(adv_string=adv_suffix).to(self.args.device),
                #     assistant_role_slice=suffix_manager._assistant_role_slice,
                #     test_prefixes=TEST_PREFIXES,
                # )
                is_success, gen_str = self.check_for_attack_success(
                    api_key=self.api_key,
                    prompt=user_prompt,
                    suffix=adv_suffix,
                )
                if j % self.args.iter == 0:
                    # 每iter轮：轮盘赌
                    unfiltered_new_adv_suffixs = autodan_sample_control(
                        control_suffixs=new_adv_suffixs,
                        score_list=score_list,
                        num_elites=num_elites,
                        batch_size=batch_size,
                        crossover=crossover,
                        num_points=num_points,
                        mutation=mutation,
                        API_key=self.api_key,
                        reference=reference,
                    )
                else:
                    # 轮盘赌换单词
                    unfiltered_new_adv_suffixs, word_dict = autodan_sample_control_hga(
                        word_dict=word_dict,
                        control_suffixs=new_adv_suffixs,
                        score_list=score_list,
                        num_elites=num_elites,
                        batch_size=batch_size,
                        crossover=crossover,
                        mutation=mutation,
                        API_key=self.api_key,
                        reference=reference,
                    )

                new_adv_suffixs = unfiltered_new_adv_suffixs

                epoch_end_time = time.time()
                epoch_cost_time = round(epoch_end_time - epoch_start_time, 2)

                print(
                    "################################\n"
                    f"当前prompt: {user_prompt}\n"
                    f"当前Epoch: {j}/{num_steps}\n"
                    f"Passed:{is_success}\n"
                    f"Loss:{current_loss.item()}\n"
                    f"Epoch花费时间:{epoch_cost_time}\n"
                    f"当前后缀:\n{best_new_adv_suffix}\n"
                    f"当前回复:\n{gen_str}\n"
                    "################################\n"
                )
                gc.collect()
                torch.cuda.empty_cache()
            info["prompt"] = user_prompt + adv_suffix
            info["response"] = gen_str
            info["attackSuccess"] = is_success
            infos.append(info)
            if is_success:
                break
        if save:
            return infos
        else:
            return info

    def check_for_attack_success(self, origin: bool = False, **kwargs):
        if origin:
            gen_str = kwargs.tokenizer.decode(
                generate(self.model, kwargs.tokenizer, kwargs.input_ids, kwargs.assistant_role_slice, gen_config=kwargs.gen_config)
            ).strip()
            uppercased_test_prefixes = [word.upper() for word in kwargs.test_prefixes]
            if len(gen_str) == 0:
                jailbroken = False
            elif gen_str.isupper():
                jailbroken = False
            else:
                jailbroken = not any([prefix in gen_str for prefix in kwargs.test_prefixes + uppercased_test_prefixes])
            return jailbroken, gen_str
        else:
            client = zhipuai.ZhipuAI(api_key=self.api_key)
            responses = client.chat.completions.create(
                model=self.attack_model,
                messages=[{"role": "user", "content": kwargs["prompt"] + kwargs["suffix"]}],
                stream=True,
            )
            chunks = ""
            for chunk in responses:
                chunks += chunk.choices[0].delta.content
            for _ in range(5):
                try:
                    json_args = judge(kwargs["prompt"], chunks, api_key=self.api_key, model_name=self.attack_model)
                    break
                except:
                    continue
            jailbroken = json_args["success"]
            return jailbroken, chunks
