import json

from torch.utils.data import Dataset
from tqdm import tqdm

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_reward",
            "description": "Get the score expected to be maximized (from 0 to 1) associated to at most 3 SMILES based on the optimization objective. Can only be called only once per conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "list[string]",
                        "description": "SMILES representations of the molecules (at most 3).",
                    },
                },
                "required": ["smiles"],
            },
        },
    }
]


def preprocess_data(
    data,
    input_template=None,
    input_key="input",
    label_key="meta",
    apply_chat_template=None,
    multimodal=False,
    use_tools=True,
    return_tokens=False,
    system_prompt: str | None = None,
):
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        tools = TOOLS if use_tools else None
        if not multimodal:
            chat = [dict(content=c["content"], role=c["role"]) for c in chat]
        if system_prompt is not None:
            with open(system_prompt, "r") as f:
                s_prompt = json.load(f)
            if chat[0]["role"] == "system":
                chat[0] = s_prompt
            else:
                chat = [s_prompt] + chat
        # Apply chat template
        try:
            try:
                chat = apply_chat_template(
                    chat, tokenize=return_tokens, add_generation_prompt=True, return_dict=multimodal, tools=tools,enable_thinking=True
                )
                print("Applied chat template with generation prompt and thinking mode.")
            except ValueError:
                chat = apply_chat_template(
                    chat, tokenize=return_tokens, add_generation_prompt=True, return_dict=multimodal, tools=tools
                )
                print("Applied chat template with generation prompt.")
        except ValueError:
            chat = apply_chat_template(chat, tokenize=return_tokens, return_dict=multimodal, tools=tools)
                print("Applied chat template.")

        if return_tokens:
            prompt = chat
            if isinstance(chat[0], list):
                mm_data = [""] * len(chat)
            else:
                mm_data = [""]
        else:
            if isinstance(chat, str):
                prompt = chat
                mm_data = [""] * len(chat)
            else:
                prompt = chat["text"]
                mm_data = chat.get("mm_data", "")

    else:
        raise NotImplementedError("apply_chat_template must be True to use this dataset.")
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    metadata = "" if label_key is None else data[label_key]
    return prompt, "|".join(mm_data), metadata


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(self, dataset, tokenizer, strategy, input_template=None, return_tokens=False) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.mm_data = []
        self.metadatas = []
        self.datasources = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, mm_data, metadata = preprocess_data(
                data,
                input_template,
                input_key,
                label_key,
                apply_chat_template,
                multimodal=self.strategy.args.multimodal,
                use_tools=self.strategy.args.use_tool_calls,
                return_tokens=return_tokens,
                system_prompt=getattr(self.strategy.args, "system_prompt", None),
            )
            self.prompts.append(prompt)
            self.mm_data.append(mm_data)
            self.metadatas.append(metadata)
            self.datasources.append(data.get("datasource", "default"))

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.datasources[idx], self.prompts[idx], self.mm_data[idx], self.metadatas[idx]
