from datetime import datetime
import yaml
import os
import json
import sys
import csv
import re
import requests
import urllib.parse
import xml.etree.ElementTree as ET
import os
import pickle
import yaml
from typing import List, Dict, Any, Optional, Union, Callable
from dotenv import load_dotenv
from openai import OpenAI
import time
import csv
import os


class DualOutput:
    """
    A class to handle simultaneous output to both the console and a file.

    Attributes:
        file (TextIO): A file object to write the output to.
        console (object): The original standard output (sys.stdout).

    Methods:
        write(message: str):
            Writes a message to both the console and the file.

        flush():
            Flushes both the console and the file buffers.
    """

    def __init__(self, file):
        """
        Initializes the DualOutput class with a file object.

        Args:
            file (TextIO): A file object where the output will be written.
        """
        self.file = file
        self.console = sys.stdout

    def write(self, message):
        """
        Writes a message to both the console and the file.

        Args:
            message (str): The message to write.
        """
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        """
        Flushes both the console and the file buffers.
        """
        self.console.flush()
        self.file.flush()


def select_llm() -> int:
    """
    Interactively prompts the user to select a LLM from a predefined list of options.

    The function displays a menu of available LLMs and waits for the user
    to input a valid choice (A-G).

    Returns:
        int: The numeric identifier corresponding to the user's selected LLM.
             (1: Qwen3-30B-A3B, 2: Mistral-Small-3.1-24B-Instruct-2503,
              3: GLM-4-32B-0414, 4: Gemma-3-27B, 5: GPT-4.1,
              6: Gemini-2.0-flash, 7: DeepSeek-V3).
    """
    llm = 100
    options = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
        "g": 7,

    }

    print("Which LLM should be used?")
    print("A: Qwen3-30B-A3B")
    print("B: Mistral-Small-3.1-24B-Instruct-2503")
    print("C: GLM-4-32B-0414")
    print("D: Gemma-3-27B")
    print("E: GPT-4.1")
    print("F: Gemini-2.0-flash")
    print("G: DeepSeek-V3")

    while llm == 100:
        choice = input("Your choice (A–G): ").lower()
        if choice in options:
            llm = options[choice]
        else:
            print("Invalid choice. Please enter a letter from A to G.")

    return llm


def load_from_file(filepath: str) -> Optional[str]:
    """
    Loads and returns the stripped content of a text file if it exists and is readable.

    This function attempts to open and read the content from the specified filepath.
    If the file exists, is readable, and contains content, the stripped content
    is printed to the console and returned. If the file does not exist, is empty,
    or an error occurs during reading, None is returned and an error message
    is printed to the console.

    Args:
        filepath (str): The full path to the file to be loaded.

    Returns:
        Optional[str]: The stripped content of the file as a string if successful,
                       otherwise None.
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                content = f.read().strip()
                if content:
                    print(f"Loaded from {filepath}: {content}")
                    return content
        except Exception as e:
            print(f"Error loading from {filepath}: {e}")
    return None


def postprocessing_response_json(string: str) -> Optional[str]:
    """
    Processes a raw LLM string response to extract and clean a JSON object string.

    It attempts to isolate the first complete JSON object within the string
    by finding the outermost curly braces. It also removes newlines and
    excess whitespace within the extracted JSON string.

    Args:
        string (str): The raw LLM response, potentially containing a JSON object
                      along with other text.

    Returns:
        Optional[str]: A string representing the cleaned JSON object if one is found.
                       Returns None if no curly braces indicating a JSON object are
                       present in the input string.
    """
    if "{" not in string or "}" not in string:
        # if no JSON object is contained, return None for applying guardrails
        return None
    string = "{" + string.split("{")[1].strip()
    string = string.split("}")[0].strip() + "}"
    while "\n" in string:
        string = string.replace("\n", " ")
    while "  " in string:
        string = string.replace("  ", " ")
    return string


def guardrails(
    original_prompt: str,
    original_input: str,
    original_response: str,
    guardrails_prompt: str,
    file_name: str,
    llm: int,
    join_folder_files: Callable[[str], str]
) -> str:
    """
    Applies a guardrail prompt to an LLM to correct or validate its previous response.

    This function sends the original prompt, input, and problematic LLM response,
    along with a specific guardrail_prompt, back to the LLM. The interaction
    (guardrail prompt and LLM's refined response) is logged to a specified file.
    The refined response is then post-processed (lowercased, quotes removed, stripped).

    Args:
        original_prompt (str): The initial prompt given to the LLM.
        original_input (str): The input data associated with the original_prompt.
        original_response (str): The LLM's response that needs correction/validation.
        guardrails_prompt (str): The system prompt instructing the LLM on how to
                                 correct/validate its previous response.
        file_name (str): The base name for the log file where the guardrail
                         interaction will be appended. A timestamp and LLM ID
                         will be prepended to this name.
        llm (int): The numeric identifier of the LLM being used.
        join_folder_files (Callable[[str], str]): A function that takes a base
                                                 filename and returns a full path,
                                                 typically by joining with a predefined
                                                 output folder.

    Returns:
        str: The post-processed (cleaned) response from the LLM after applying
             the guardrail prompt.
    """
    messages_history = list()
    messages_history.append({"role": "system", "content": guardrails_prompt})
    prompt = "Original task:\n" + original_prompt + "\n\n" + original_input + "\n\nResponse by LLM agent:\n" + original_response
    response = query_llm_agent(prompt, messages_history, llm)
    f = open(join_folder_files(add_timestamp_and_llm(llm) + file_name), "a", encoding="utf-8")
    f.write("\nPrompt:\n " + prompt)
    f.write("\nResponse:\n" + response + "\n\n\n")
    f.close()
    response = response.lower().replace("'", "").replace('"', "").strip()
    return response


def add_timestamp_and_llm(llm: int) -> str:
    """
    Generates a string prefix for filenames, incorporating the current timestamp
    and the identifier of the LLM being used.

    The format is "YYYY-MM-DD_HH-MM_llm_X_", where X is the LLM identifier.

    Args:
        llm (int): The numeric identifier of the language model.

    Returns:
        str: A formatted string suitable for prepending to filenames,
             e.g., "2023-10-27_14-30_llm_1_".
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M") + "_llm_" + str(llm) + "_"


def list_to_comma_separated_string(in_list: List[Any]) -> str:
    """
    Converts a list of any items into a single comma-separated string.

    Each item in the list is converted to its string representation.

    Args:
        in_list (List[Any]): The list to be converted.

    Returns:
        str: A string with elements from the list joined by ", ".
             Returns an empty string if the input list is empty.
    """
    string = ', '.join(map(str, in_list))
    return string


def query_llm_agent(prompt: str, messages_history: List[Dict[str, str]], llm: int, json_schema: Optional[str] = None) -> str:
    """
    Sends a prompt to a specified Large Language Model (LLM) API and retrieves its response.

    This function handles the configuration for various LLMs, including local
    models accessed via an OpenAI-compatible API (like LM Studio) and remote
    proprietary APIs (GPT, Gemini, DeepSeek). It supports structured JSON output
    if a json_schema is provided and the LLM supports it. Includes error handling
    for API key issues and basic retry/timeout logic implicit in API clients or via sleep.

    Args:
        prompt (str): The user's query or instruction to be sent to the LLM.
        messages_history (List[Dict[str, str]]): A list of message objects
            representing the conversation history, in the format expected by
            the OpenAI API (e.g., [{"role": "system", "content": "..."}, ...]).
        llm (int): An integer identifier specifying which LLM to use.
            (1: Qwen3, 2: Mistral-Small, 3: GLM-4, 4: Gemma-3, 5: GPT-4.1,
             6: Gemini-2.0-flash, 7: DeepSeek-V3).
        json_schema (Optional[str], optional): A JSON string representing the
            schema for the expected JSON output. If provided and supported by the LLM,
            the LLM will attempt to return a response conforming to this schema.
            Defaults to None, in which case a standard text completion is requested.

    Returns:
        str: The LLM's response content as a string. Returns an empty string
             if an API error occurs or if the response is refused by the model
             (e.g., due to content filters or schema incompatibility).

    """

    load_dotenv()
    if llm == 5:
        api_key = os.getenv("GPT_KEY")
    elif llm == 6:
        api_key = os.getenv("GEMINI_25_KEY")
    elif llm == 7:
        api_key = os.getenv("DEEPSEEK_KEY")
    else: # for gemini and all other llms, use gemini key free tier
        api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the respective environment variable.")

    if llm == 1:
        prompt = "/no_think\n" + prompt  # just for qwen to avoid reasoning
        model = "Qwen3-30B-A3B"
        client = OpenAI(
            api_key=api_key,
            base_url="https://25cd-134-109-211-240.ngrok-free.app/v1"
        )
    elif llm == 2:
        model = "Mistral-Small-3.1-24B-Instruct-2503-UD-Q4_K_XL"
        client = OpenAI(
            api_key=api_key,
            base_url="https://d093-2a02-2454-88a2-aa00-99ac-6b82-95c3-adb5.ngrok-free.app/v1"
        )
    elif llm == 3:
        model = "GLM-4-32B-0414-UD-Q4_K_XL"
        client = OpenAI(
            api_key=api_key,
            base_url="https://4627-134-109-211-27.ngrok-free.app/v1"
        )
    elif llm == 4:
        model = "gemma-3-27b-it-UD-Q4_K_XL"
        client = OpenAI(
            api_key=api_key,
            base_url="https://d11a-134-109-211-209.ngrok-free.app/v1"
        )

    elif llm == 5:
        model = "gpt-4o-mini"
        organization = os.getenv("ORGANIZATION")
        if not organization:
            raise ValueError(
                "Organization for OpenAI access not found. Please set the ORGANIZATION environment variable.")

        client = OpenAI(
            organization=organization,
            api_key=api_key
        )
    elif llm == 6:
        model = "gemini-2.0-flash"
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/"
        )
    elif llm == 7:
        model = "deepseek-chat"
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    else:
        model = "gemini-2.0-flash"
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/"
        )
    messages_history.append({"role": "user", "content": prompt})
    try:
        #if llm == 6 or llm > 7:
            #time.sleep(4)
            # waiting time was added to not surpass the RPM (requests per minute) limit of the free tier of gemini-2.0-flash
            # https://ai.google.dev/pricing?hl=en

        if json_schema:

            json_schema = json.loads(json_schema)

            response = client.chat.completions.create(
                model=model,
                messages=messages_history,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": json_schema
                }
                )
            print(response)

            # check if it was refused:
            message = response.choices[0].message
            if message.refusal:
                print("refusal in querying:")
                print(message.refusal)
                return ""
            else:
                return message.content.strip()

        else:
            print("in else")

            response = client.chat.completions.create(
                    model=model,
                    messages=messages_history,
                    temperature=0
                )
            print("in else 2")
            print(response)
            response = response.choices[0].message.content.strip()
            # only for gwen3: remove think tags at the beginning of response
            if llm == 1:
                response = response.replace("<think>", "").replace("</think>", "").lstrip()

        return response

    except Exception as e:
        print(f"Error querying LLM API: {e}")
        return ""


def get_cso_csv() -> str:
    """
    Returns the filename of the CSO CSV file.

    This function centralizes the reference to the specific version of the CSO
    dataset used in the project, facilitating easier updates if the version changes.

    Returns:
        str: The filename "CSO.3.3.csv".
    """
    return "CSO.3.3.csv"


def load_prompts_from_yaml(file_name: str, join_folder_files: Callable[[str], str]) -> List[str]:
    """
    Loads a list of prompts from a specified YAML file.

    Each item in the YAML file's root list is considered a separate prompt.
    Prompts are stripped of leading/trailing whitespace. Includes error handling
    for file not found and YAML parsing errors.

    Args:
        file_name (str): The base name of the YAML file (e.g., "prompts.yaml").
        join_folder_files (Callable[[str], str]): A function that takes the base
            filename and returns a full path, typically by joining it with a
            predefined directory for configuration files.

    Returns:
        List[str]: A list of prompt strings. Returns an empty list if the file
                   cannot be found, cannot be parsed, or an unexpected error occurs.
    """

    prompts = list()
    try:
        with open(join_folder_files(file_name), 'r') as file:
            prompts_yaml = yaml.safe_load(file)
            prompts = [prompt.strip() for prompt in prompts_yaml]
        return prompts

    except FileNotFoundError as fnf_error:
        print(f"Error: The file '{file_name}' was not found. {fnf_error}")
        return prompts

    except yaml.YAMLError as yaml_error:
        print(f"Error: Failed to parse yaml file '{file_name}'. {yaml_error}")
        return prompts

    except Exception as e:
        print(f"An unexpected error while handling the yaml file {file_name} occurred: {e}")
        return prompts


def postprocessing_list_response(string: str) -> Union[str, List[Any]]:
    """
    Processes a raw LLM string response to extract a string representation of a list.

    It attempts to find the first occurrence of a list-like structure (enclosed in '[...]')
    within the input string. If found, it extracts this substring and removes
    excess internal whitespace. If no list structure is found, it returns an empty list.

    Note: This function returns the list as a *string*, not a parsed Python list object,
    unless no list is found, in which case an empty Python list `[]` is returned.

    Args:
        string (str): The raw LLM response, potentially containing a list.

    Returns:
        Union[str, List[Any]]: A string representation of the first found list
                               (e.g., "[item1, item2, item3]") with normalized
                               whitespace, if a list is found. Otherwise, returns
                               an empty Python list `[]`.
    """
    start = string.find("[")
    end = string.find("]", start)

    if start == -1 or end == -1:
        return list()
    # Everything inside the brackets
    lst = string[start:end + 1]
    without_whitespace = " ".join(lst.split())
    return without_whitespace
