from datasets import load_dataset
from datasets.distributed import split_dataset_by_node


def get_laser_sft_dataset(
    path,
    split,
    tokenizer,
    rank,
    world_size,
    max_length=None,
    **kwargs,
):
    del kwargs
    dataset = load_dataset(path=path, name="default", split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def process(sample):
        seq_token = tokenizer.encode(
            sample["prompt"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["prompt"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["prompt", "ref_output_tokens_count", "length_range", "answer"])
    if max_length is not None:
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)
    return dataset


def get_laser_rl_dataset(
    path,
    split,
    rank,
    world_size,
    tokenizer=None,
    max_length=None,
    **kwargs,
):
    del kwargs
    dataset = load_dataset(path=path, name="default", split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def process(sample):
        messages = [{"role": "user", "content": sample["prompt"]}]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["prompt", "ref_output_tokens_count", "length_range"])
    if max_length is not None and tokenizer is not None:

        def filter_length(sample):
            if tokenizer.chat_template is not None:
                input_ids = tokenizer.apply_chat_template(
                    sample["messages"],
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                return input_ids.shape[-1] <= max_length

            content = "\n".join(message["content"] for message in sample["messages"])
            return len(tokenizer.encode(content)) <= max_length

        dataset = dataset.filter(filter_length)
    return dataset
