"""
Author: Morph Labs <support@morph.so>
"""

import rich
import rich.console
from alive_progress import alive_bar
import time
from dataclasses import dataclass, field, asdict
import fire
import os
import requests
from tqdm import tqdm
from llama_cpp.llama import Llama
import enum
from typing import List, Optional
import transformers

def morph_splash():
    _splash = """
------------------------------------------------------
      __  ___                 __
     /  |/  /___  _________  / /_
    / /|_/ / __ \\/ ___/ __ \\/ __ \\
   / /  / / /_/ / /  / /_/ / / / /
  /_/  /_/\\____/_/  / .___/_/ /_/
                 __/_/
                / __ \\_________ _   _____  _____
               / /_/ / ___/ __ \\ | / / _ \\/ ___/
              / ____/ /  / /_/ / |/ /  __/ /
             /_/   /_/   \\____/|___/\\___/_/

------------------------------------------------------
"""

    def stream_string(string):
        for char in string:
            print(char, end="", flush=True)
            time.sleep(0.0008)
            # print('\r', end='')

    stream_string(_splash)


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>", BOS, EOS]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

def format_llama_conversation(turns: List[str]) -> str:
    """
    Format llama conversation for Morph Prover.
    """
    assert len(turns) > 0

    turn_texts = []

    for turn_idx in range(0, len(turns), 2):
        prompt = turns[turn_idx]
        if turn_idx == 0:
            prompt = " ".join(
                [
                    B_SYS,
                    "You are a helpful assistant.",
                    E_SYS,
                    prompt,
                ]
            )

        response = turns[turn_idx + 1] if turn_idx + 1 < len(turns) else ""

        turn_texts.append(
            " ".join(
                [
                    B_INST,
                    prompt,
                    E_INST,
                    response,
                ]
            )
        )

    return f" {EOS} {BOS} ".join(turn_texts)

@dataclass
class Message:
    role: str
    content: str


@dataclass
class ChatState:
    messages: List[Message] = field(default_factory=list)

    def compile_conversation(self):
        return format_llama_conversation([m.content for m in self.messages] + [""])

def _ensure_model(model_path: Optional[str] = None):
    model_dir = os.path.expandvars("$HOME/.morph")
    model_file = "gguf-model-Q8_0.gguf"
    model_path = model_path or os.path.join(model_dir, model_file)

    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)

        # Download the model
        url = "https://huggingface.co/morph-labs/morph-prover-v0-7b-gguf/resolve/main/gguf-model-Q8_0.gguf"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(model_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                progress_bar.update(len(chunk))
                file.write(chunk)

        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

    return model_path


def main(model_path: Optional[str] = None, gpu: bool = False):
    console = rich.console.Console(); print = console.print
    
    morph_splash()
    extra_kwargs = dict()
    if gpu:
        extra_kwargs["n_gpu_layers"] = 1000
        extra_kwargs["n_threads"] = 1

    print("loading model")
    with alive_bar(title='Loading model...', spinner='bubbles') as bar:
        model = Llama(
            model_path=_ensure_model(model_path),
            n_ctx=4096,
            verbose=False,
            **extra_kwargs
        )
        bar()

    chat_state = ChatState()

    print("starting chat")
    
    while True:
        user_message = input("user: ")
        chat_state.messages.append(Message(role="user", content=user_message))
        # model_response_stream = model.create_completion(chat_state.compile_conversation(), max_tokens=1024, temperature=0.25, top_p=0.9, top_k=64, stream=True)
        model_response_stream = model.create_chat_completion(messages=[asdict(m) for m in chat_state.messages], max_tokens=1024, temperature-0.25, top_p=0.9, top_k=64, stream=True)
        print("morph-prover-v0-7b: ", end="")
        resp = ""
        for chunk in model_response_stream:
            try:
                next_text_chunk = chunk["choices"][0]["content"]
                print(next_text_chunk, end="")
                resp += next_text_chunk
                bar()
            except Exception as e:
                print(f"caught {e=}")
                continue
        print()
        chat_state.messages.append(Message(role="assistant", content=resp))



if __name__ == "__main__":
    fire.Fire(main)
