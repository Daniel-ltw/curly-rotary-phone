import json
import random
from openai import Client
from datetime import date
import os
import numpy as np
import soundfile as sf
from outetts import HFModelConfig_v2, InterfaceHF, GenerationConfig

device = "cpu"

def random_pause(sample_rate, min_duration=0.5, max_duration=1.5):
    silence_duration = random.uniform(min_duration, max_duration)
    silence = np.zeros(int(silence_duration * sample_rate))
    return silence

def produce_audio(name, script):
    model_config = HFModelConfig_v2(
        model_path="OuteAI/OuteTTS-0.3-500M",
        tokenizer_path="OuteAI/OuteTTS-0.3-500M"
    )
    interface = InterfaceHF(model_version="0.3", cfg=model_config)

    lea_speaker = interface.load_default_speaker(name="en_female_1")
    jon_speaker = interface.load_default_speaker(name="en_male_2")

    audio = []

    for i, sentence in enumerate(script):
        voice = sentence["voice"]
        text = sentence["text"]
        print(f"{i + 1}/{len(script)}: Creating audio with {voice}: {text}")

        if voice == "Lea":
            generation_config = GenerationConfig(
                speaker=lea_speaker,
                temperature=0.7,
                repetition_penalty=1.1,
                text=text
            )
        else:
            generation_config = GenerationConfig(
                speaker=jon_speaker,
                temperature=0.3,
                repetition_penalty=1.1,
                text=text
            )

        output = interface.generate(config=generation_config)
        audio.append(output.audio.squeeze())
        # Add random silence after each sentence
        audio.append(random_pause(output.sr))

    if len(audio) > 1:
        # Concatenate all audio parts
        audio = np.concatenate(audio)

        # Save the generated audio to file
        sf.write(f"{name}.wav", audio, 44100)

def get_script(filename):
    with open(filename, "r") as f:
        content = f.read()

    client = Client(base_url="http://localhost:1234/v1", api_key="lm-studio")
    response = client.chat.completions.create(
        model="lm_studio/qwen2.5-7b-instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experienced podcast script writer."
                    "\nYour task is to write a script for a lengthy podcast based on the news of the day. "
                    "\nThe script should be in a conversational style between Claudia and David. "
                    "\nWhen it is Claudia's lines, make sure to follow the format, the line must start with `** Claudia: <line>\\n`. "
                    "\nWhen it is David's lines, make sure to follow the format, the line must start with `** David: <line>\\n`. "
                    "\n\n"
                    "** IMPORTANT **"
                    "\n\t- The hosts should cover every single detail of each news summary provided by the user."
                    "\n\t- The hosts should delve more into each news item, discussing or debating about it. "
                    "\n\t- The hosts should rely the news in a way that is both informative and entertaining, and make it engaging and interesting for the listeners. "
                    "\n\t- DO NOT be selective about the news items. Cover all of them. "
                    "\n\t- Ensure that the script is between 13000 and 16000 tokens. "
                    "\n\t- The script should elaborate on the news items so that it exceeds 50 lines. "
                    "\n\n"
                    "The user will provide you with the summaries of the news of the day. "
                )
            },
            {"role": "user", "content": content}
        ],
        max_tokens=16000,
    )
    return response.choices[0].message.content

def organize_script(script):
    script_json = []
    speaker = None
    for line in script.split("\n"):
        if not line.startswith("** Claudia:") and not line.startswith("** David:"):
            continue

        line = line.replace("**", "").strip()
        json_speech = {}
        if line.startswith("Claudia:"):
            json_speech["voice"] = "Lea"
            line = line.replace("Claudia:", "")
        elif line.startswith("David") and line.replace("David:", "").strip() != "":
            json_speech["voice"] = "Jon"
            line = line.replace("David:", "")
        else:
            continue

        json_speech["text"] = line.strip()
        script_json.append(json_speech)

    return script_json

def main():
    today = date.today()
    filename = f"./output/{today.strftime('%Y-%m-%d')}.md"
    if not os.path.exists(filename):
        print(f"File {filename} does not exist")
        return

    script = get_script(filename)
    print("script: ", script)
    script_json = organize_script(script)
    print("Number of lines: ", len(script_json))
    print("Number of tokens: ", len(json.dumps(script_json)))
    produce_audio(today.strftime('%Y-%m-%d'), script_json)
    print("Podcast generated")


if __name__ == "__main__":
    main()
