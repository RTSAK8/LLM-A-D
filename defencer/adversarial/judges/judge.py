from defencer.adversarial.judges.keyword import GCGKeywordMatchingJudge, KeywordMatchingJudge
from defencer.adversarial.judges.openai_policy_judge import OpenAIPolicyGPTJudge
from defencer.adversarial.judges.pair_judge import PAIRGPTJudge
from defencer.adversarial.judges.quality_judge import QualityGPTJudge
from defencer.adversarial.judges.no_judge import NoJudge


judge_dict={
    "quality": QualityGPTJudge,
    "openai_policy": OpenAIPolicyGPTJudge,
    "pair": PAIRGPTJudge,
    "no-judge": NoJudge,
    "matching": KeywordMatchingJudge,
    "gcg_matching": GCGKeywordMatchingJudge
}


def load_judge(judge_name, goal, **kwargs):
    if judge_name in judge_dict:
        judge = judge_dict[judge_name](goal, **kwargs)
    else:
        judge_name, model_name = judge_name.split("@")
        if judge_name in judge_dict:
            judge = judge_dict[judge_name](goal, model_name, **kwargs)
        else:
            raise NotImplementedError
    return judge
