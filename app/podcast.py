import json
import random
from openai import Client
from datetime import date
import os
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

def random_pause(sample_rate, min_duration=1.0, max_duration=3.0):
    silence_duration = random.uniform(min_duration, max_duration)
    silence = np.zeros(int(silence_duration * sample_rate))
    return silence

def produce_audio(name, script):
    kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")

    audio = []

    for i, sentence in enumerate(script[:10]):
        voice = sentence["voice"]
        text = sentence["text"]
        print(f"{i + 1}/{len(script)}: Creating audio with {voice}: {text}")

        samples, sample_rate = kokoro.create(
            text,
            voice=voice,
            lang="en-gb",
        )
        audio.append(samples)
        # Add random silence after each sentence
        audio.append(random_pause(sample_rate))

    # Concatenate all audio parts
    audio = np.concatenate(audio)

    # Save the generated audio to file
    sf.write(f"{name}.wav", audio, sample_rate)

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
                    "\n\n"
                    "** IMPORTANT **"
                    "\n\t- The hosts should cover every single detail of each news summary provided by the user."
                    "\n\t- The hosts should delve more into each news item, discussing or debating about it. "
                    "\n\t- The hosts should rely the news in a way that is both informative and entertaining, and make it engaging and interesting for the listeners. "
                    "\n\t- DO NOT be selective about the news items. Cover all of them. "
                    "\n\t- Ensure that the script is between 13000 and 16000 tokens. "
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
    for line in script.split("\n"):
        if not line.startswith("**"):
            continue

        line = line.replace("**", "")
        json_speech = {}
        if line.startswith("Claudia"):
            json_speech["voice"] = "bf_emma"
            line = line.replace("Claudia:", "")
        elif line.startswith("David"):
            json_speech["voice"] = "bm_lewis"
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
    script_json = organize_script(script)
    print("Number of lines: ", len(script_json))
    print("Number of tokens: ", len(json.dumps(script_json)))
    produce_audio(today.strftime('%Y-%m-%d'), script_json)
    print("Podcast generated")


if __name__ == "__main__":
    main()
