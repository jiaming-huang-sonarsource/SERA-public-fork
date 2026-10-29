import copy
import json
import re

def remove_think_message(content):
    message_lines = content.splitlines()
    cleaned_lines = []
    for line in message_lines:
        if "<think>" not in line and "</think>" not in line:
            cleaned_lines.append(line)
    content = "\n".join(cleaned_lines)
    return content

def reformat_think_message(content):
    """
    Ensure that content follows Qwen3 format of newlines before/after every <think> tag
    """
    think_content = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    if len(think_content) != 1:
        return None
    new_think_section = "\n".join(["<think>", think_content[0].strip(), "</think>"])
    try:
        new_content = re.sub(r"<think>.*?</think>", lambda m: new_think_section, content, flags=re.DOTALL)
    except Exception as e:
        return None
    return new_content

def parse_text_indexed(text):
    """
    Parses text using indexed capture groups.
    The <think> section is optional.
    """
    # Regex with standard (indexed) capture groups
    pattern = re.compile(
        r"^(<think>.*?</think>)?(.*?)(<tool_call>.*?</tool_call>)$",
        re.DOTALL
    )
    
    match = pattern.match(text.strip())
    
    if match:
        # Returns a tuple of all captured groups
        return match.groups()
    return None

##############################
# processing fns

def add_train_key(dataset):
    dataset = copy.deepcopy(dataset)
    for data in dataset:
        for i, message in enumerate(data["messages"]):
            if message["role"] != "assistant":
                message["train"] = False
            else:
                message["train"] = True
    return dataset

def reformat_assistant_message(dataset, mode: str):
    print("reformatting think")
    assert mode in ["keep_only_think", "keep_only_non_think"]
    dataset = copy.deepcopy(dataset)
    rft_dataset = []
    for data in dataset:
        reformat_success = True
        for i, message in enumerate(data["messages"]):
            if "<think>" in message["content"]:
                message["content"] = reformat_think_message(message["content"])
                if message["content"] is None:
                    reformat_success = False
        if reformat_success:
            rft_dataset.append(data)

    rft_select_dataset = []
    for data in rft_dataset:
        fail = False
        for i, message in enumerate(data["messages"]):
            if message["role"] == "assistant":
                parsed_text = parse_text_indexed(message['content']) # TODO: Make this work with other tool call formats
                if parsed_text is None:
                    fail = True
                    break
                think_content, commentary, tool_call = parsed_text
                if mode == "keep_only_think":
                    if think_content is None or re.findall(r"<think>(.*?)</think>", think_content, flags=re.DOTALL)[0].strip() == "":
                        # Add thinks around commentary if no think block is present
                        think_content = "\n".join(["<think>", commentary.strip(), "</think>"])
                elif mode == "keep_only_non_think":
                    if commentary.strip() != "" or think_content is None:
                        # Use commentary to reason as long as it exists OR think is empty
                        think_content = "\n".join(["<think>", commentary.strip(), "</think>"])
                message["content"] = "\n\n".join([think_content, tool_call])
        if not fail:
            rft_select_dataset.append(data)
    print("remaining instances:", len(rft_select_dataset))
    return rft_select_dataset

XML_STR_REPLACES = ["old_str", "new_str", "file_text"] # TODO: Create copyright 
def transform_traj_xml(traj: dict, system_prompt: str, add_think: bool = False) -> dict:
    def tool_call_to_action(tool_calls):
        actions = []
        if tool_calls is None:
            return []
        for tool_call in tool_calls:
            action = [f"<function={tool_call['function']['name']}>"]
            arguments = json.loads(tool_call["function"]["arguments"])
            for k, v in arguments.items():
                a = f"<parameter={k}>{v}</parameter>"
                if k in XML_STR_REPLACES:
                    a = f"<parameter={k}>\n{v}\n</parameter>"
                action.append(a)
            action.append("</function>")
            actions.append("\n".join(action))
        return actions

    new_traj = []
    messages = traj["history"] # traj["trajectory"][-1]["query"][:-1]
    for message in messages:
        role = message["role"] if message["role"] != "tool" else "user"
        if message["role"] == "assistant":
            if message["content"] == "Exit due to cost limit":
                content = (
                    "Since we have successfully fixed the issue and verified it works, "
                    + "let's submit the changes:\n\n"
                    + "<function=submit>\n</function>"
                )
            else:
                if add_think:
                    content = message['content'].strip()
                    content = "\n".join(["<think>", content, "</think>"])
                    message['content'] = content
                action = "\n".join(tool_call_to_action(message["tool_calls"]))
                content = f"{message['content']}\n\n{action}"
        elif message["role"] == "system":
            content = system_prompt
        else:
            if isinstance(message["content"], list):
                assert len(message["content"]) == 1
                content = message["content"][0]["text"]
            elif isinstance(message["content"], str):
                content = message["content"]
            else:
                raise ValueError(f"Message type not recognized: {type(message)}")
        new_traj.append({"role": role, "content": content})
    return {"messages": new_traj}

def transform_traj_hermes(traj: dict, system_prompt: str, add_think: bool = False) -> dict:
    def tool_call_to_action(tool_calls):
        actions = []
        if tool_calls is None:
            return []
        for tool_call in tool_calls:
            # {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed"}}
            action = ["<tool_call>"]
            tool_call = {"name": tool_call['function']['name'], "arguments": json.loads(tool_call["function"]["arguments"])}
            action.append(json.dumps(tool_call))
            action.append("</tool_call>")
            actions.append("\n".join(action))
        return actions
    def tool_response(tool_response):
        response = ["<tool_response>", tool_response, "</tool_response>"]
        return "\n".join(response)
    new_traj = []
    messages = traj["history"] # traj["trajectory"][-1]["query"][:-1]
    for message in messages:
        # print(message)
        if message["role"] == "tool":
            if isinstance(message["content"], list):
                assert len(message["content"]) == 1
                message["content"][0]["text"] = tool_response(message["content"][0]["text"])
            elif isinstance(message["content"], str):
                message["content"] = tool_response(message["content"])
            role = "user"
        else:
            role = message["role"]
        # Add processed message to new trajectory
        if message["role"] == "assistant":
            if message["content"] == "Exit due to cost limit":
                if not add_think:
                    content = (
                        "Since we have successfully fixed the issue and verified it works, "
                        + "let's submit the changes:\n\n"
                        + "<tool_call>\n{\"name\": \"submit\", \"arguments\": {}}\n</tool_call>"
                    )
                else:
                    content = (
                        "<think>\nSince we have successfully fixed the issue and verified it works, "
                        + "let's submit the changes:\n</think>\n\n"
                        + "<tool_call>\n{\"name\": \"submit\", \"arguments\": {}}\n</tool_call>"
                    ) 
            else:
                if message["tool_calls"] is not None:
                    if add_think:
                        content = message['content'].strip()
                        content = "\n".join(["<think>", content, "</think>"])
                        message['content'] = content
                    action = "\n".join(tool_call_to_action(message["tool_calls"]))
                    content = f"{message['content']}\n\n{action}"
                else:
                    content = convert_xml_hermes(message["content"], message["role"], think=add_think)
                    if not content:
                        return None
        elif message["role"] == "system":
            content = system_prompt
        else:
            if isinstance(message["content"], list):
                assert len(message["content"]) == 1
                content = message["content"][0]["text"]
            elif isinstance(message["content"], str):
                content = message["content"]
            else:
                raise ValueError(f"Message type not recognized: {type(message)}")
        new_traj.append({"role": role, "content": content})
    return {"messages": new_traj}

def transform_traj_raw(traj: dict, system_prompt: str, add_think: bool = False):
    return {"messages": traj["history"]}
