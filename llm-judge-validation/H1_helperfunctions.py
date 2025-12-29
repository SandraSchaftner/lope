import os
import json
import re
from typing import Dict, Any, List, Optional

NUM_PAPERS = 8


def join_folder_files(file_name: str) -> str:
    """
    Join the given file name with the "H1 files" directory.

    Args:
        file_name (str): The name of the file to be joined with the folder path.

    Returns:
        str: Full path to the file inside the "H1 files" directory.
    """
    return os.path.join("H1 files", file_name)


def join_folder_results(file_name: str) -> str:
    """
    Join the given file name with the "H1 results" directory.

    Args:
        file_name (str): The name of the file to be joined with the folder path.

    Returns:
        str: Full path to the file inside the "H1 results" directory.
    """
    return os.path.join("H1 results", file_name)


def setup_evaluation_structure_for_skg(skg_name: str = "allmac") -> Dict[str, Dict[str, Dict[str, None]]]:
    """
    Creates an initialized nested dictionary structure for storing evaluation
    scores for a single SKG type (e.g., "allmac", "cs-kg", "orkg").

    The structure is `eval_json[skg_name][criterion_name][paper_key] = None`.
    - `criterion_name` is one of the 8 quality criteria.
    - `paper_key` is a string like "1_skgname" or "11_skgname" (where 1-8 are
      individual papers and 11 represents a combined/aggregated TTL file).

    Args:
        skg_name (str, optional): The name of the SKG for which the structure
                                  is being created. Defaults to "allmac".

    Returns:
        Dict[str, Dict[str, Dict[str, None]]]: The initialized evaluation dictionary
                                               for the specified SKG.
    """
    eval_json = {skg_name: {}}
    criteria = get_criteria_list()
    paper_ids_for_eval = list(range(1, NUM_PAPERS + 1))  # 1 to 8
    paper_ids_for_eval.append(11)  # for combined TTL

    skg_data = eval_json[skg_name]
    for crit in criteria:
        skg_data[crit] = {}
        for paper_id in paper_ids_for_eval:
            key = f"{paper_id}_{skg_name}"  # e.g., "1_allmac", "11_allmac"
            skg_data[crit][key] = None
    return eval_json


def get_criteria_list() -> List[str]:
    """
    Returns a predefined list of the eight quality criteria names used for SKG evaluation.

    Returns:
        List[str]: A list containing: "semantic_accuracy", "syntactic_accuracy",
                   "conciseness", "completeness", "consistency", "interoperability",
                   "reusability", "understandability".
    """
    criteria = [
        "semantic_accuracy",
        "syntactic_accuracy",
        "conciseness",
        "completeness",
        "consistency",
        "interoperability",
        "reusability",
        "understandability"
    ]
    return criteria


def get_3_skgs_list() -> List[str]:
    """
    Returns a predefined list of three SKG names relevant to the evaluation:
    "allmac", "cs-kg", "orkg".

    Returns:
        List[str]: The list of three SKG names.
    """
    skgs = ["allmac", "cs-kg", "orkg"]
    return skgs


def get_2_skgs_list() -> List[str]:
    """
    Returns a predefined list of two SOTA SKG names relevant to the evaluation:
    "cs-kg", "orkg".

    Returns:
        List[str]: The list of two SOTA SKG names.
    """
    skgs = ["cs-kg", "orkg"]
    return skgs


def get_llm_id_from_filename(filename: str) -> Optional[str]:
    """
    Extracts an LLM numeric identifier from a filename string.

    Args:
        filename (str): The filename string to parse.

    Returns:
        Optional[str]: The extracted LLM ID as a string if the pattern is found.
                       Returns None otherwise.
    """
    match = re.search(r"_llm_(\d+)_", filename)
    if match:
        return match.group(1)
    return None


def generate_judge_json_schema(llm: int) -> str:
    """
    Dynamically generates a JSON schema string for guiding the LLM judge's
    evaluation response.

    Args:
        llm (int): The numeric identifier of the LLM judge. This is used to
                   tailor the integer constraint method in the schema (enum vs. min/max).

    Returns:
        str: A string representing the JSON schema for the LLM judge's output.
    """

    criteria = get_criteria_list()
    keys = criteria.copy()
    keys.append("justification")

    base_schema_intro = """
                        {
                          "name": "RDF_Turtle_evaluation",
                          "schema": {
                            "type": "object",
                            "properties": {
                        """
    properties_string = ""
    for c in criteria:
        if llm == 5:  # gpt uses enum, others minimum and maximum
            prop_detail = f'''
                                  "{c}": {{
                                    "type": "integer",
                                    "description": "An integer value between 1 and 5 (inclusive) for the evaluation of the criterion {c} according to the given instructions.",
                                    "enum": [1, 2, 3, 4, 5]
                                  }},'''
        else:
            prop_detail = f'''
                                  "{c}": {{
                                    "type": "integer",
                                    "description": "An integer value between 1 and 5 (inclusive) for the evaluation of the criterion {c} according to the given instructions.",
                                    "minimum": 1,
                                    "maximum": 5
                                  }},'''
        properties_string += prop_detail

    justification_prop = f'''
                                  "justification": {{
                                    "type": "string",
                                    "description": "A short justification text why you assigned which value to which criterion. Focus on the main one or two reasons for each criterion."
                                  }}'''
    properties_string += justification_prop

    schema_outro = f"""
                            }},
                                "required": {json.dumps(keys)},
                                "additionalProperties": false
                              }},
                              "strict": true
                            }}
                            """
    return base_schema_intro + properties_string + schema_outro


def is_complete_evaluation(json_object: Optional[Dict[str, Any]]) -> bool:
    """
    Validates if an LLM judge's evaluation output (as a dictionary) is complete
    and adheres to the expected format.

    Args:
        json_object (Optional[Dict[str, Any]]): The dictionary parsed from the
                                                 LLM judge's JSON response.

    Returns:
        bool: True if the evaluation object is complete and valid, False otherwise.
    """
    if not isinstance(json_object, dict):
        return False
    required_keys = get_criteria_list()
    required_keys.append("justification")

    for key in required_keys:
        if key not in json_object:
            return False
        if key != "justification":
            if not isinstance(json_object[key], int) or not (1 <= json_object[key] <= 5):
                return False
    return True


def normalize_json_object(json_object: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Normalizes an LLM judge's evaluation JSON object by converting keys to
    lowercase and attempting to cast criteria values to integers.

    Args:
        json_object (Optional[Dict[str, Any]]): The dictionary parsed from the
                                                 LLM judge's JSON response.

    Returns:
        Optional[Dict[str, Any]]: A new dictionary with normalized keys and values.
                                  Returns None if the input was not a dictionary.
    """
    if not isinstance(json_object, dict):
        return None
    normalized = {}
    for key, value in json_object.items():
        key_lower = key.lower()
        if key_lower != "justification":
            try:
                normalized_value = int(value)
            except (ValueError, TypeError):
                normalized_value = 0
        else:
            normalized_value = str(value)
        normalized[key_lower] = normalized_value
    return normalized


def get_context_for_id(paper_id_for_eval: int,
                       dois_file_names_map: Dict[int, str],
                       publications_data: List[Dict[str, str]]) -> str:
    """
    Retrieves the title and abstract of a publication to provide context for the LLM judge.

    Args:
        paper_id_for_eval (int): The numeric identifier of the paper (e.g., 1, 2, ..., 8).
                                 If this ID is not found in `dois_file_names_map` (e.g.,
                                 for a combined file ID like 11), an empty string is returned.
        dois_file_names_map (Dict[int, str]): A dictionary mapping integer paper IDs
                                             to their base filenames (without extension).
        publications_data (List[Dict[str, str]]): A list of dictionaries, where each
            dictionary represents a publication and is expected to have "file_name",
            "title", and "abstract" keys.

    Returns:
        str: A formatted string containing "Title: ...\nAbstract:\n...\nRDF Turtle file:\n".
             Returns an empty string if no context can be found (e.g., for combined
             files or if the paper ID/filename is not found).
    """
    base_filename = dois_file_names_map.get(paper_id_for_eval)
    if not base_filename:
        return ""

    for pub in publications_data:
        if pub.get("file_name") == base_filename:  # 'file_name' in title_and_abstract_8_papers.json
            title = pub.get("title", "No title available")
            abstract = pub.get("abstract", "No abstract available")
            return f"Title: {title}\nAbstract:\n{abstract}\nRDF Turtle file:\n"
    return ""
