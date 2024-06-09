from .adapters import SparkAdapter, ZhiPuAdapter
from fastchat.model.model_adapter import register_model_adapter
from fastchat.conversation import Conversation, register_conv_template

# 注册新的模型
register_model_adapter(SparkAdapter)
register_conv_template(
    Conversation(
        name="spark",
        system_message="You are a helpful assistant.",
        roles=["user", "assistant"],
        offset=0,
    )
)
register_model_adapter(ZhiPuAdapter)
register_conv_template(
    Conversation(
        name="zhipu",
        system_message="You are a helpful assistant.",
        roles=["user", "assistant"],
        offset=0,
    )
)
