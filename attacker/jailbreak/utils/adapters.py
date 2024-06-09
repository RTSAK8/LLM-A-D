from fastchat.conversation import Conversation
from fastchat.model.model_adapter import BaseModelAdapter, get_conv_template


class SparkAdapter(BaseModelAdapter):
    """The model adapter for Spark"""

    def match(self, model_path: str):
        return model_path == "spark"

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("spark")


class ZhiPuAdapter(BaseModelAdapter):
    """Zhipu AI"""

    def match(self, model_path: str):
        return model_path == "zhipu"

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        raise NotImplementedError()

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("zhipu")
