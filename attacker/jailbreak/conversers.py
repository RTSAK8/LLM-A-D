from attacker.jailbreak import common
from attacker.jailbreak.config import *
from attacker.jailbreak.utils.language_models import GPT, LanguageModel, Zhipu
from typing import Tuple


def load_attack_and_target_models(args):
    # Load attack model and tokenizer
    attack_llm = AttackLLM(
        model_name=args.attack_model,
        max_n_tokens=args.attack_max_n_tokens,
        max_n_attack_attempts=args.max_n_attack_attempts,
        temperature=ATTACK_TEMP,  # init to 1
        top_p=ATTACK_TOP_P,  # init to 0.9
        api_key=args.api_key,
    )
    preloaded_model = None
    # if args.attack_model == args.target_model:
    #     print("攻击模型和目标模型使用同一模型")
    #     preloaded_model = attack_llm.model
    target_llm = TargetLLM(
        model_name=args.target_model,
        max_n_tokens=args.target_max_n_tokens,
        temperature=TARGET_TEMP,  # init to 0
        top_p=TARGET_TOP_P,  # init to 1
        preloaded_model=preloaded_model,
        api_key=args.api_key,
    )
    return attack_llm, target_llm


class AttackLLM:
    """
    Base class for attacker language models.

    Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """

    def __init__(
        self,
        model_name: str,
        max_n_tokens: int,
        max_n_attack_attempts: int,
        temperature: float,
        top_p: float,
        api_key: str,
    ):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name, api_key)

        if "vicuna" in model_name or "llama" in model_name:
            self.model.extend_eos_tokens()

    def get_attack(self, conv_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model.
        Only valid outputs in proper JSON format are returned. If an output isn't generated
        successfully after max_n_attack_attempts, it's returned as None.

        Parameters:
        - conv_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.

        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """

        assert len(conv_list) == len(prompts_list), "conversations数量和prompts数量不匹配"

        batch_size = len(conv_list)
        indices_to_regenerate = list(range(batch_size))
        valid_outputs = [None] * batch_size

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(conv_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            # if "gpt" in self.model_name:
            full_prompts.append(conv.to_openai_api_messages())

        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs
            outputs_list = self.model.batched_generate(
                full_prompts_subset, max_n_tokens=self.max_n_tokens, temperature=self.temperature, top_p=self.top_p
            )

            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]

                attack_dict, json_str = common.extract_json(full_output)

                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    if "gpt" in conv_list[orig_index].name:
                        conv_list[orig_index].update_last_message(json_str)  # Update the conversation with valid generation
                    else:
                        conv_list[orig_index].append_message(conv_list[orig_index].roles[1], json_str)
                else:
                    new_indices_to_regenerate.append(orig_index)

            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate

            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs


class TargetLLM:
    """
    Base class for target language models.

    Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """

    def __init__(
        self,
        model_name: str,
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        preloaded_model: object,
        api_key: str,
    ):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        self.api_key = api_key
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name, api_key)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def get_response(self, prompts_list):
        batch_size = len(prompts_list)
        conv_list = [common.conv_template(self.template) for _ in range(batch_size)]
        full_prompts = []
        for conv, prompt in zip(conv_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                # Openai does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                full_prompts.append([{"role": conv.roles[0], "content": prompt}])

        outputs_list = self.model.batched_generate(full_prompts, max_n_tokens=self.max_n_tokens, temperature=self.temperature, top_p=self.top_p)
        return outputs_list


def load_indiv_model(model_name, api_key) -> Tuple[LanguageModel, str]:
    model_path, template = get_model_path_and_template(model_name)
    if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-0125-preview"]:
        llm = GPT(model_name, api_key)
    elif model_name in ["zhipu", "glm-4"]:
        llm = Zhipu("glm-4", api_key)
    else:
        raise ValueError(f"Model {model_name} not Found.")
    return llm, template


def get_model_path_and_template(model_name):
    full_model_dict = {
        "gpt-4": {"path": "gpt-4", "template": "gpt-4"},
        "gpt-3.5-turbo": {"path": "gpt-3.5-turbo", "template": "gpt-3.5-turbo"},
        "gpt-4-0125-preview": {"path": "gpt-4-0125-preview", "template": "gpt-4-0125-preview"},
        "spark": {"path": "spark", "template": "spark"},
        "glm-4": {"path": "glm-4", "template": "glm-4"},
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template
