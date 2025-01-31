import json
import random
from openai import Client
from datetime import date
import os
import numpy as np
import soundfile as sf
from kokoro import KPipeline

def random_pause(sample_rate, min_duration=0.3, max_duration=1.0):
    silence_duration = random.uniform(min_duration, max_duration)
    silence = np.zeros(int(silence_duration * sample_rate))
    return silence

def produce_audio(name, script):
    pipeline = KPipeline(lang_code='b')

    audio = []

    for i, sentence in enumerate(script):
        voice = sentence["voice"]
        text = sentence["text"]
        print(f"{i + 1}/{len(script)}: Creating audio with {voice}: {text}")

        generator = pipeline(
            text,
            voice=voice,
        )
        for _graphemes, _phonemes, samples in generator:
            audio.append(samples)
            # Add random silence after each sentence
            audio.append(random_pause(24000))

    # Concatenate all audio parts
    audio = np.concatenate(audio)

    # Save the generated audio to file
    sf.write(f"{name}.wav", audio, 24000)

def get_script(filename):
    with open(filename, "r") as f:
        content = f.read()

    client = Client(base_url="http://localhost:1234/v1", api_key="lm-studio")
    response = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-14b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an experienced podcast script writer."
                    "\nYour task is to write a script for a lengthy podcast based on the news of the day. "
                    "\nThe script should be in a conversational style between Claudia and David. "
                    "\nWhen it is Claudia's lines, make sure to follow the format, Claudia's line must start with `** Claudia: <line>\\n`. "
                    "\nWhen it is David's lines, make sure to follow the format, David's line must start with `** David: <line>\\n`. "
                    "\n\n"
                    "** IMPORTANT **"
                    "\n\t- The hosts should cover every single detail of each news summary provided by the user."
                    "\n\t- The hosts should delve more into each news item, discussing or debating about it. "
                    "\n\t- The hosts should rely the news in a way that is both informative and entertaining, and make it engaging and interesting for the listeners. "
                    "\n\t- DO NOT be selective about the news items. Cover all of them. "
                    "\n\t- Ensure that the script is between 13000 and 16000 tokens. "
                    "\n\t- Each news item should be AT LEAST 3-7 lines long, so it could be elaborated or expanded on. "
                    "\n\t- When covering a new news item, MAKE SURE to MENTION the new item title. "
                    "\n\t- The output should ONLY contain the script, no other text. "
                    "\n\t- NEVER generate your own news items, ALWAYS use the news items provided by the user. "
                    "\n\n"
                    "<example output>\n"
                    "** Claudia: This is 'The Daily' for today. We're going to be talking about the latest news for the day. Here we have David, how are you doing?\n"
                    "** David: I'm doing great, thanks for asking. We seem to have a list of items to cover today. Let's start with the first one. \n"
                    "** Claudia: I heard about the new AI model that was released yesterday. What do you think about it?\n"
                    "** David: I think it's a game changer. It's going to revolutionize the way we interact with AI. What do you think?\n"
                    "** Claudia: I agree with you. It's amazing how quickly AI is evolving. Do you think it will replace human jobs anytime soon?\n"
                    "** David: I don't think so. AI is still far from being able to fully replace human jobs. But it's definitely changing the way we work. What do you think?\n"
                    "..............................\n"
                    "** Claudia: That wraps up today’s news. It’s been an insightful discussion, David. Thanks for your input.\n"
                    "** David: Always a pleasure, Claudia. Stay informed, everyone, and we’ll see you next time.\n"
                    "</example output>"
                    "\n\n"
                    "The user will provide you with the summaries of the news of the day. Ensure to cover all of the summaries. "
                )
            },
            {"role": "user", "content": (
                "Here are the summaries of the news of the day:\n\n"
                "<summaries>\n"
                f"{content}\n"
                "</summaries>"
            )}
        ],
        max_tokens=16000,
    )
    return response.choices[0].message.content

def organize_script(script):
    script_json = []
    for line in script.split("\n"):
        if not line.startswith("**"):
            continue

        line = line.replace("**", "").strip()
        json_speech = {}
        if line.startswith("Claudia"):
            json_speech["voice"] = "bf_emma"
            line = line.replace("Claudia:", "").strip()
            if len(line) < 1:
                raise ValueError("Claudia's line is empty")
        elif line.startswith("David"):
            json_speech["voice"] = "bm_george"
            line = line.replace("David:", "").strip()
            if len(line) < 1:
                raise ValueError("David's line is empty")
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
    print("Script: ", script)
    script_json = organize_script(script)
    print("Number of lines: ", len(script_json))
    print("Number of tokens: ", len(json.dumps(script_json)))
    produce_audio(today.strftime('%Y-%m-%d'), script_json)
    print("Podcast generated")


if __name__ == "__main__":
    main()
