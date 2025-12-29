"""
IMPORTANT:
This script is just needed if any LLM judge evaluation should be conducted beside the one implemented in H1_pipeline.py

H1_LLM_judge.py - LLM-based Evaluation of CS-KG and ORKG Outputs.

This script orchestrates the evaluation of RDF Turtle (TTL) files generated for two SOTA SKGs, CS-KG and ORKG,
using a selected LLM as the "LLM Judge". The evaluation is based on eight predefined quality criteria.

Files needed (typically in "H1 files" or as specified by `join_folder_files`):
-   orkg_all_rdf_ttl.pickle (Cached ORKG TTL data for the 8 papers)
-   cskg_all_rdf_ttl.pickle (Cached CS-KG TTL data for the 8 papers)
-   dois_file_names_2.json (Mapping paper IDs to DOIs and filenames)
-   H1_prompts.yaml (Prompts for the LLM judge)
-   title_and_abstract_8_papers.json (Context for individual paper evaluations)

External Services:
-   CS-KG SPARQL endpoint (if pickle not found)
-   ORKG API (if pickle not found)
-   LLM API (for the selected LLM judge)
"""
import os
import json
import pickle
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, DC, OWL
import requests
from typing import Dict, Any, List, Optional
from helperfunctions import (
    select_llm, add_timestamp_and_llm, query_llm_agent,
    load_prompts_from_yaml, guardrails, postprocessing_response_json
)
from H1_helperfunctions import (
    get_criteria_list,
    get_2_skgs_list,
    join_folder_results,
    join_folder_files,
    generate_judge_json_schema,
    get_context_for_id,
    normalize_json_object,
    is_complete_evaluation
)


def setup_eval_json_local() -> Dict[str, Any]:
    """
    Creates and initializes the specific JSON structure used to store the LLM
    Judge's evaluation scores for CS-KG and ORKG.

    Returns:
        Dict[str, Any]: The initialized nested dictionary for storing evaluation scores.
                        Example path: `eval_json["cs-kg"]["semantic_accuracy"]["1_cs-kg"]`.
    """
    eval_json: Dict[str, Any] = {}
    skgs: List[str] = get_2_skgs_list()  # ["cs-kg", "orkg"]
    criteria: List[str] = get_criteria_list()
    # Paper IDs 1 through 8 for individual papers, and 11 for the combined file
    publication_ids: List[int] = list(range(1, 9))  # 1 to 8
    publication_ids.append(11)  # for combined TTL

    for skg_name in skgs:
        eval_json[skg_name] = {}
        for criterion_name in criteria:
            eval_json[skg_name][criterion_name] = {}
            for pub_id in publication_ids:
                #  "1_cs-kg", "11_orkg"
                score_key = f"{pub_id}_{skg_name}"
                eval_json[skg_name][criterion_name][score_key] = None
    return eval_json


def get_all_ttl(all_rdf_ttl_list: List[Dict[str, Any]]) -> str:
    """
    Merges multiple RDF Turtle (TTL) strings, each associated with a paper/entity,
    into a single aggregated TTL string.

    Each item in `all_rdf_ttl_list` is expected to be a dictionary with an 'id'
    (for identification, used in error reporting) and an 'rdf_ttl' key containing
    the Turtle content. The function initializes an `rdflib.Graph`, binds a
    comprehensive set of common namespaces, parses each valid TTL string into
    the graph, and then serializes the combined graph back to a Turtle string.
    Warnings are printed for TTL content that fails to parse.

    Args:
        all_rdf_ttl_list (List[Dict[str, Any]]): A list of dictionaries, where each
            dictionary should contain at least an "rdf_ttl" key with the Turtle
            string, and an "id" key for context in case of parsing errors.

    Returns:
        str: A single Turtle string representing the merged RDF graph.
    """
    g = Graph()
    DC_TERMS = Namespace("http://purl.org/dc/terms/")
    RDFS_NS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    ORKGP = Namespace("https://orkg.org/property/")
    DBPEDIA = Namespace("http://dbpedia.org/resource/")
    WIKIDATA = Namespace("http://www.wikidata.org/entity/")
    CSO = Namespace("https://cso.kmi.open.ac.uk/topics/")
    FABIO = Namespace("http://purl.org/spar/fabio/")
    FOAF = Namespace("http://xmlns.com/foaf/0.1/")
    SCHEMA = Namespace("http://schema.org/")
    NFDICORE = Namespace("https://ise-fizkarlsruhe.github.io/nfdicore/")
    IAO = Namespace("http://purl.obolibrary.org/obo/")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    OWL_NS = Namespace("http://www.w3.org/2002/07/owl#")
    CSKG_RES_NS = Namespace(
        "https://w3id.org/cskg/resource/")
    CSKG_ONT_NS = Namespace("https://w3id.org/cskg/ontology/")
    ORKG_NS = Namespace("https://orkg.org/orkg/")

    g.bind("dc", DC_TERMS, replace=True, override=True)
    g.bind("rdfs", RDFS_NS, replace=True, override=True)
    g.bind("orkgp", ORKGP, replace=True, override=True)
    g.bind("dbpr", DBPEDIA, replace=True, override=True)
    g.bind("wikidata", WIKIDATA, replace=True, override=True)
    g.bind("cso", CSO, replace=True, override=True)
    g.bind("fabio", FABIO, replace=True, override=True)
    g.bind("foaf", FOAF, replace=True, override=True)
    g.bind("schema", SCHEMA, replace=True, override=True)
    g.bind("nfdicore", NFDICORE, replace=True, override=True)
    g.bind("iao", IAO, replace=True, override=True)
    g.bind("skos", SKOS, replace=True, override=True)
    g.bind("owl", OWL_NS, replace=True, override=True)
    g.bind("cskg", CSKG_RES_NS, replace=True, override=True)
    g.bind("cskgo", CSKG_ONT_NS, replace=True, override=True)
    g.bind("orkg", ORKG_NS, replace=True, override=True)

    for mapping_item in all_rdf_ttl_list:
        rdf_ttl_content = mapping_item.get("rdf_ttl")
        if rdf_ttl_content:
            try:
                g.parse(data=rdf_ttl_content, format="turtle", publicID="https://allmac-project.com/default_graph/")
            except Exception as e:
                paper_id = mapping_item.get('id', 'Unknown ID')
                print(f"Warning: Could not parse TTL content for paper ID {paper_id}. Error: {e}")

    combined_ttl_str = g.serialize(format="turtle")
    return combined_ttl_str


def cskg_convert_to_ttl(cskg_json_data: Dict[str, Any]) -> str:
    """
    Converts CS-KG data, typically from a SPARQL JSON response format, into an
    RDF graph serialized in Turtle (TTL).

    Args:
        cskg_json_data (Dict[str, Any]): The JSON object parsed from a CS-KG SPARQL
                                        query response. Expected to have a structure
                                        like `{'results': {'bindings': [...]}}`.

    Returns:
        str: A string containing the RDF graph serialized in Turtle format.
    """
    bindings = cskg_json_data.get('results', {}).get('bindings', [])

    CSKG_RESOURCE = Namespace("https://w3id.org/cskg/resource/")
    CSKG_ONTOLOGY = Namespace("https://w3id.org/cskg/ontology#")

    g = Graph()
    g.bind("cskg", CSKG_RESOURCE)
    g.bind("cskgo", CSKG_ONTOLOGY)
    g.bind("rdfs", RDFS)  # from rdflib.namespace
    g.bind("dc", DC)  # from rdflib.namespace
    g.bind("owl", OWL)  # from rdflib.namespace

    entities_to_query_sameas = set()

    for binding_item in bindings:
        subj_uri = URIRef(binding_item['sub']['value'])
        pred_uri = URIRef(binding_item['prop']['value'])
        obj_uri = URIRef(binding_item['obj']['value'])
        paper_uri = URIRef(binding_item['paper']['value'])
        doi_val = binding_item['doi']['value']
        doi_lit = Literal(doi_val, datatype="http://www.w3.org/2001/XMLSchema#anyURI")

        g.add((subj_uri, pred_uri, obj_uri))
        g.add((subj_uri, CSKG_ONTOLOGY.wasDerivedFrom, paper_uri))
        g.add((paper_uri, CSKG_ONTOLOGY.doi, doi_lit))

        if 'subLabel' in binding_item:
            g.add((subj_uri, RDFS.label, Literal(binding_item['subLabel']['value'], lang="en")))
        if 'propLabel' in binding_item:
            g.add((pred_uri, RDFS.label, Literal(binding_item['propLabel']['value'], lang="en")))
        if 'objLabel' in binding_item:
            g.add((obj_uri, RDFS.label, Literal(binding_item['objLabel']['value'], lang="en")))
        if 'title' in binding_item:
            g.add((paper_uri, RDFS.label, Literal(binding_item['title']['value'], lang="en")))

        entities_to_query_sameas.add(subj_uri)
        entities_to_query_sameas.add(obj_uri)

    for entity_uri_to_check in entities_to_query_sameas:
        sameas_results = query_cskg_sameas(str(entity_uri_to_check))
        for sameas_binding in sameas_results:
            linked_entity_uri = URIRef(sameas_binding['same_entity']['value'])
            g.add((entity_uri_to_check, OWL.sameAs, linked_entity_uri))

    return g.serialize(format='turtle')


def query_cskg(doi_str: str) -> Optional[Dict[str, Any]]:
    """
    Queries the CS-KG SPARQL endpoint to retrieve triples and associated metadata
    (labels, paper title) related to a specific publication DOI.

    Args:
        doi_str (str): The DOI of the publication (e.g., "10.1000/xyz123").
                       The function formats this into the URI expected by CS-KG.

    Returns:
        Optional[Dict[str, Any]]: The parsed JSON response from the SPARQL endpoint
                                  if the query is successful and returns data.
                                  Returns None if the request fails (e.g., network
                                  error, non-200 status).
    """
    endpoint_url = "http://192.167.149.12:9000/sparql/"
    doi_uri_in_query = f"https%3A//doi.org/{doi_str}"

    sparql_query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX cskg: <https://w3id.org/cskg/resource/>
    PREFIX cskg-ont: <https://w3id.org/cskg/ontology#>
    PREFIX provo: <http://www.w3.org/ns/prov#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?sub ?prop ?obj ?paper ?doi ?title ?subLabel ?propLabel ?objLabel
    FROM <https://w3id.org/cskg>
    WHERE {{
        ?paper cskg-ont:doi "{doi_uri_in_query}"^^<http://www.w3.org/2001/XMLSchema#anyURI> .
        ?triple rdf:subject ?sub ;
                rdf:predicate ?prop ;
                rdf:object ?obj ;
                provo:wasDerivedFrom ?paper .
        BIND(?paper AS ?entity_with_doi) # Use BIND for clarity if needed, or directly use ?paper
        ?entity_with_doi cskg-ont:doi ?doi . # Re-confirm DOI association if structure is complex

        OPTIONAL {{ ?paper rdfs:label ?title . }}
        OPTIONAL {{ ?sub rdfs:label ?subLabel . }}
        OPTIONAL {{ ?prop rdfs:label ?propLabel . }}
        OPTIONAL {{ ?obj rdfs:label ?objLabel . }}
    }}
    LIMIT 10000
    """
    request_headers = {"Accept": "application/sparql-results+json"}
    try:
        response = requests.post(endpoint_url, data={"query": sparql_query}, headers=request_headers,
                                 timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"CS-KG SPARQL query failed for DOI {doi_str}. Error: {e}")
        return None


def query_cskg_sameas(entity_uri_str: str) -> List[Dict[str, Any]]:
    """
    Queries the CS-KG SPARQL endpoint to find all `owl:sameAs` links for a given
    CS-KG entity URI.

    Args:
        entity_uri_str (str): The full URI of the CS-KG entity.

    Returns:
        List[Dict[str, Any]]: A list of bindings from the SPARQL result, where each
                              binding contains a 'same_entity' key with the URI of
                              an equivalent entity. Returns an empty list if no
                              links are found or if the query fails.
    """
    endpoint_url = "http://192.167.149.12:9000/sparql/"
    sparql_query = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT ?same_entity
    FROM <https://w3id.org/cskg>
    WHERE {{
        <{entity_uri_str}> owl:sameAs ?same_entity .
    }}
    """
    request_headers = {"Accept": "application/sparql-results+json"}
    try:
        response = requests.post(endpoint_url, data={"query": sparql_query}, headers=request_headers, timeout=10)
        response.raise_for_status()
        return response.json().get('results', {}).get('bindings', [])
    except requests.exceptions.RequestException as e:
        print(f"CS-KG SPARQL sameAs query failed for entity {entity_uri_str}. Error: {e}")
        return []


def get_rdf_ttl_cskg(doi_file_entry: Dict[str, Any]) -> Optional[str]:
    """
    Orchestrates fetching CS-KG data for a publication (identified by DOI)
    and converting it into an RDF Turtle (TTL) string.

    Args:
        doi_file_entry (Dict[str, Any]): A dictionary containing "file_name" (used for
            naming the saved JSON file) and "doi" (the publication's DOI).

    Returns:
        Optional[str]: The RDF Turtle string if CS-KG data is successfully fetched
                       and converted. Returns None otherwise.
    """
    json_filename = doi_file_entry["file_name"] + ".json"
    doi_value = doi_file_entry["doi"]

    cskg_sparql_result = query_cskg(doi_value)
    if cskg_sparql_result:
        cskg_json_output_dir = join_folder_results("cskg_json/")
        os.makedirs(cskg_json_output_dir, exist_ok=True)
        try:
            filepath = os.path.join(cskg_json_output_dir, json_filename)
            with open(filepath, 'w', encoding='utf-8') as f_json:
                json.dump(cskg_sparql_result, f_json, indent=4)
            print(f"  CS-KG JSON data saved to: {filepath}")
        except Exception as e_save:
            print(f"  Error writing CS-KG JSON to file {json_filename}: {e_save}")

        rdf_ttl_output = cskg_convert_to_ttl(cskg_sparql_result)
        return rdf_ttl_output
    return None


def get_rdf_ttl_orkg(doi_file_entry: Dict[str, Any]) -> Optional[str]:
    """
    Orchestrates fetching ORKG data for a publication (identified by DOI)
    and converting it into an RDF Turtle (TTL) string.

    Args:
        doi_file_entry (Dict[str, Any]): A dictionary containing "doi" (the
                                      publication's DOI). The "file_name" key is
                                      present but not directly used by this function.

    Returns:
        Optional[str]: The RDF Turtle string if ORKG data is successfully fetched
                       and converted. Returns None otherwise.
    """
    doi_value = doi_file_entry["doi"]
    orkg_statements_list = query_orkg(doi_value)

    if orkg_statements_list:
        rdf_ttl_output = orkg_convert_to_ttl(orkg_statements_list, doi_value)
        return rdf_ttl_output
    return None


def get_orkg_paper_by_doi(doi_str: str) -> Optional[Dict[str, Any]]:
    """
    Fetches metadata for a specific paper from the ORKG API using its DOI.

    Args:
        doi_str (str): The DOI of the paper.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the paper's metadata
                                  (e.g., ORKG ID, title, list of contributions)
                                  if found. Returns None if the paper is not found
                                  or an API error occurs.
    """
    api_url = f"https://incubating.orkg.org/api/papers?doi={doi_str}"
    request_headers = {"Accept": "application/vnd.orkg.paper.v2+json"}
    try:
        response = requests.get(api_url, headers=request_headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get("content"):
            return data["content"][0]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ORKG paper by DOI {doi_str}: {e}")
    return None


def get_same_as_links(orkg_entity_id: str) -> Optional[List[str]]:
    """
    Fetches `owl:sameAs` links for a given ORKG entity ID by querying the
    ORKG statements API for statements with the predicate "SAME_AS".

    Args:
        orkg_entity_id (str): The ORKG ID of the entity (resource or property).

    Returns:
        Optional[List[str]]: A list of URI strings that are `owl:sameAs` the input
                             entity. Returns None if no such links are found or an
                             API error occurs. Returns an empty list if the API call
                             is successful but no content is returned.
    """
    api_url = f"https://incubating.orkg.org/api/statements?subject_id={orkg_entity_id}&predicate_id=SAME_AS"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        same_as_uri_list = [
            stmt["object"]["label"]
            for stmt in data.get("content", [])
            if stmt.get("object") and "label" in stmt["object"]
        ]
        return same_as_uri_list if same_as_uri_list else None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ORKG sameAs links for entity {orkg_entity_id}: {e}")
    return None


def get_orkg_label(orkg_entity_id: str) -> Optional[str]:
    """
    Fetches the `rdfs:label` for an ORKG entity (resource or predicate) given its ID.

    It first attempts to fetch the entity as a resource. If not found (or on error),
    it then attempts to fetch it as a predicate.

    Args:
        orkg_entity_id (str): The ORKG ID (e.g., "R123" or "P456").

    Returns:
        Optional[str]: The label string if found. Returns None if the entity is not
                       found as either type or if API errors occur.
    """
    base_api_url = "https://incubating.orkg.org/api"
    for entity_path_segment in [f"resources/{orkg_entity_id}", f"predicates/{orkg_entity_id}"]:
        full_api_url = f"{base_api_url}/{entity_path_segment}"
        try:
            response = requests.get(full_api_url, timeout=10)
            if response.status_code == 200:
                return response.json().get("label")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching ORKG label for {orkg_entity_id} from {full_api_url}: {e}")
    return None


def orkg_convert_to_ttl(orkg_statements: List[Dict[str, Any]], doi_str: str) -> str:
    """
    Converts a list of ORKG property statements for a paper (identified by DOI)
    into an RDF graph serialized in Turtle (TTL) format.

    Args:
        orkg_statements (List[Dict[str, Any]]): A list of dictionaries, where each
            represents a statement. Expected keys: "contribution_id", "property"
            (predicate ORKG ID), "values" (list of object ORKG IDs).
        doi_str (str): The DOI of the paper, used to fetch its primary metadata.

    Returns:
        str: A string containing the RDF graph serialized in Turtle format. Returns
             an empty string if essential paper metadata for the DOI cannot be retrieved.
    """
    paper_metadata = get_orkg_paper_by_doi(doi_str)
    if not paper_metadata:
        print(f"Could not retrieve ORKG paper metadata for DOI {doi_str}. TTL will be incomplete or empty.")
        return ""

    paper_orkg_id = paper_metadata["id"]
    paper_title_str = paper_metadata["title"]

    ORKG_VOCAB = Namespace("https://orkg.org/orkg/")

    g = Graph()
    g.bind("orkg", ORKG_VOCAB, replace=True, override=True)
    g.bind("rdfs", RDFS, replace=True, override=True)
    g.bind("dc", DC, replace=True, override=True)
    g.bind("owl", OWL, replace=True, override=True)

    paper_main_uri = ORKG_VOCAB[paper_orkg_id]
    g.add((paper_main_uri, DC.title, Literal(paper_title_str)))

    entities_processed_for_labels_sameas = set()

    for statement_entry in orkg_statements:
        contrib_id = statement_entry["contribution_id"]
        predicate_id = statement_entry["property"]
        object_id_list = statement_entry["values"]

        for obj_id in object_id_list:
            g.add((ORKG_VOCAB[contrib_id], ORKG_VOCAB[predicate_id], ORKG_VOCAB[obj_id]))

            for entity_id_to_detail in [contrib_id, predicate_id, obj_id]:
                if entity_id_to_detail not in entities_processed_for_labels_sameas:
                    label_str = get_orkg_label(entity_id_to_detail)
                    if label_str:
                        g.add((ORKG_VOCAB[entity_id_to_detail], RDFS.label, Literal(label_str)))

                    same_as_uri_list = get_same_as_links(entity_id_to_detail)
                    if same_as_uri_list:
                        for linked_uri_str in same_as_uri_list:
                            if linked_uri_str.startswith("http://") or linked_uri_str.startswith("https://"):
                                g.add((ORKG_VOCAB[entity_id_to_detail], OWL.sameAs, URIRef(linked_uri_str)))

                    entities_processed_for_labels_sameas.add(entity_id_to_detail)

    return g.serialize(format="turtle")


def query_orkg(doi_str: str) -> List[Dict[str, Any]]:
    """
    Queries the ORKG API to retrieve all property-based statements for all
    contributions associated with a paper identified by its DOI.

    Args:
        doi_str (str): The DOI of the publication.

    Returns:
        List[Dict[str, Any]]: A list of statement dictionaries. Returns an empty
                              list if the paper or its contributions are not found,
                              or if API errors occur.
    """
    paper_metadata = get_orkg_paper_by_doi(doi_str)
    if not paper_metadata:
        print(f"No ORKG paper found for DOI: {doi_str}")
        return []

    contributions_list = paper_metadata.get("contributions", [])
    if not contributions_list:
        print(f"No contributions found for ORKG paper DOI: {doi_str}")
        return []

    print(f"  Found {len(contributions_list)} contributions for DOI {doi_str}. Fetching details...")

    all_paper_statements = []
    for contribution_item in contributions_list:
        contribution_id_str = contribution_item["id"]
        contribution_api_url = f"https://incubating.orkg.org/api/contributions/{contribution_id_str}"
        contrib_request_headers = {"Accept": "application/vnd.orkg.contribution.v2+json"}

        try:
            contrib_response = requests.get(contribution_api_url, headers=contrib_request_headers, timeout=10)
            contrib_response.raise_for_status()
            contribution_data = contrib_response.json()
        except requests.exceptions.RequestException as e:
            print(f"    Error fetching ORKG contribution {contribution_id_str}: {e}")
            continue

        properties_dict = contribution_data.get("properties", {})

        for predicate_orkg_id, object_orkg_ids_list in properties_dict.items():
            all_paper_statements.append({
                "contribution_id": contribution_id_str,
                "property": predicate_orkg_id,
                "values": object_orkg_ids_list
            })

    return all_paper_statements


def setup_output_folders() -> None:
    """
    Creates necessary output subdirectories within the main "H1 results" folder
    if they do not already exist.

    The created folders are: "orkg_ttl", "cskg_ttl", "cskg_json",
    "llm_judge_cs-kg_orkg", and "all_justifications".
    """
    output_folder_names = [
        "orkg_ttl", "cskg_ttl", "cskg_json",
        "llm_judge_cs-kg_orkg",
        "all_justifications"
    ]
    for folder_name_item in output_folder_names:
        full_folder_path = join_folder_results(folder_name_item)
        os.makedirs(full_folder_path, exist_ok=True)


def save_ttl_file(rdf_ttl_content: str, base_filename_no_ext: str, output_subfolder_name: str) -> None:
    """
    Saves a given RDF Turtle (TTL) string to a `.ttl` file within a specified
    subfolder of the main "H1 results" directory.

    Args:
        rdf_ttl_content (str): The RDF data in Turtle format to be saved.
        base_filename_no_ext (str): The base name for the output file (e.g., "paper1").
                                    The ".ttl" extension will be appended.
        output_subfolder_name (str): The name of the subfolder within "H1 results"
                                     (e.g., "cskg_ttl") where the file will be saved.
    """
    ttl_actual_filename = base_filename_no_ext + ".ttl"

    full_output_filepath = join_folder_results(os.path.join(output_subfolder_name, ttl_actual_filename))

    os.makedirs(os.path.dirname(full_output_filepath), exist_ok=True)

    try:
        with open(full_output_filepath, 'w', encoding="utf-8", errors="ignore") as f_ttl:
            f_ttl.write(rdf_ttl_content)
        print(f"  TTL data saved to: {full_output_filepath}")
    except Exception as e_write:
        print(f"  Error writing TTL to file {full_output_filepath}: {e_write}")


def save_final_evaluation_json(evaluation_data: Dict[str, Any], judge_llm_id_used: int) -> None:
    """
    Saves the final LLM Judge evaluation data (a nested dictionary of scores)
    to a JSON file.

    Args:
        evaluation_data (Dict[str, Any]): The dictionary containing the evaluation scores.
        judge_llm_id_used (int): The numeric identifier of the LLM that performed
                                 the judging.
    """
    target_output_folder = join_folder_results("llm_judge_cs-kg_orkg")
    os.makedirs(target_output_folder, exist_ok=True)

    output_filename = add_timestamp_and_llm(judge_llm_id_used, prefix="") + "judge_cs-kg_orkg.json"
    full_output_filepath = os.path.join(target_output_folder, output_filename)

    try:
        with open(full_output_filepath, 'w', encoding='utf-8', errors="ignore") as f_out:
            json.dump(evaluation_data, f_out, indent=4)
        print(f"LLM Judge evaluation data successfully saved to: {full_output_filepath}")
    except Exception as e_final_save:
        print(f"Error writing final LLM Judge evaluation JSON to {full_output_filepath}: {e_final_save}")


def main():
    llm_judge_id_selected = select_llm()
    print(f"Using LLM ID {llm_judge_id_selected} as the LLM Judge.")

    setup_output_folders()
    try:
        with open(join_folder_files("dois_file_names_2.json"), "r", encoding="utf-8") as f_dois:
            doi_metadata_list = json.load(f_dois)
        doi_to_filename_map = {item["id"]: item["file_name"] for item in doi_metadata_list}
    except FileNotFoundError:
        print("Error: 'dois_file_names_2.json' not found. This file is crucial. Exiting.")
        return
    except json.JSONDecodeError:
        print("Error: Could not decode 'dois_file_names_2.json'. Exiting.")
        return

    try:
        with open(join_folder_files("title_and_abstract_8_papers.json"), "r", encoding="utf-8") as f_context:
            paper_context_list = json.load(f_context)
    except FileNotFoundError:
        print("Warning: 'title_and_abstract_8_papers.json' not found. Context for LLM judge will be limited.")
        paper_context_list = []
    except json.JSONDecodeError:
        print("Warning: Could not decode 'title_and_abstract_8_papers.json'. Using empty context.")
        paper_context_list = []

    cskg_individual_ttls = []
    cskg_pickle_path = join_folder_files("cskg_all_rdf_ttl.pickle")
    if os.path.exists(cskg_pickle_path):
        with open(cskg_pickle_path, "rb") as f_pickle_cskg:
            cskg_individual_ttls = pickle.load(f_pickle_cskg)
        print("Loaded CS-KG TTLs from pickle cache.")
    else:
        print("Pickle cache for CS-KG not found. Querying CS-KG API...")
        for doi_entry_item in doi_metadata_list:
            ttl_str = get_rdf_ttl_cskg(doi_entry_item)
            if ttl_str:
                save_ttl_file(ttl_str, doi_entry_item["file_name"], "cskg_ttl")
                cskg_individual_ttls.append({"id": doi_entry_item["id"], "rdf_ttl": ttl_str})
        if cskg_individual_ttls:
            with open(cskg_pickle_path, "wb") as f_pickle_cskg:
                pickle.dump(cskg_individual_ttls, f_pickle_cskg)
            print("Saved fetched CS-KG TTLs to pickle cache.")

    cskg_combined_ttl_str = get_all_ttl(cskg_individual_ttls)
    if cskg_combined_ttl_str:
        save_ttl_file(cskg_combined_ttl_str, "combined", "cskg_ttl")

    orkg_individual_ttls = []
    orkg_pickle_path = join_folder_files("orkg_all_rdf_ttl.pickle")
    if os.path.exists(orkg_pickle_path):
        with open(orkg_pickle_path, "rb") as f_pickle_orkg:
            orkg_individual_ttls = pickle.load(f_pickle_orkg)
        print("Loaded ORKG TTLs from pickle cache.")
    else:
        print("Pickle cache for ORKG not found. Querying ORKG API...")
        for doi_entry_item in doi_metadata_list:
            ttl_str = get_rdf_ttl_orkg(doi_entry_item)
            if ttl_str:
                save_ttl_file(ttl_str, doi_entry_item["file_name"], "orkg_ttl")
                orkg_individual_ttls.append({"id": doi_entry_item["id"], "rdf_ttl": ttl_str})
        if orkg_individual_ttls:
            with open(orkg_pickle_path, "wb") as f_pickle_orkg:
                pickle.dump(orkg_individual_ttls, f_pickle_orkg)
            print("Saved fetched ORKG TTLs to pickle cache.")

    orkg_combined_ttl_str = get_all_ttl(orkg_individual_ttls)
    if orkg_combined_ttl_str:
        save_ttl_file(orkg_combined_ttl_str, "combined", "orkg_ttl")

    final_judge_evaluation_output = setup_eval_json_local()

    skg_iteration_map = {0: "cs-kg", 1: "orkg"}
    list_of_all_individual_ttls = [cskg_individual_ttls, orkg_individual_ttls]
    list_of_all_combined_ttls = [cskg_combined_ttl_str, orkg_combined_ttl_str]

    judge_prompts = load_prompts_from_yaml("H1_prompts.yaml", join_folder_files)
    judge_json_schema_str = generate_judge_json_schema(llm_judge_id_selected)

    for skg_map_idx, skg_specific_individual_ttls in enumerate(list_of_all_individual_ttls):
        current_skg_being_judged = skg_iteration_map[skg_map_idx]
        print(f"\n--- LLM Judge evaluating INDIVIDUAL papers for: {current_skg_being_judged.upper()} ---")

        for paper_ttl_mapping in skg_specific_individual_ttls:
            paper_unique_id = paper_ttl_mapping["id"]
            individual_ttl_str = paper_ttl_mapping.get("rdf_ttl")

            print(f"  Judging SKG: {current_skg_being_judged}, Paper ID: {paper_unique_id}")

            if not individual_ttl_str:
                print(f"    Skipping Paper ID {paper_unique_id} for {current_skg_being_judged} (no TTL content).")
                continue

            paper_context_str = get_context_for_id(paper_unique_id, doi_to_filename_map, paper_context_list)
            llm_input_content = paper_context_str + individual_ttl_str

            judged_json_object = None
            response_from_llm = ""
            max_judge_attempts = 5

            for attempt_num in range(max_judge_attempts):
                print(f"    Attempt {attempt_num + 1}/{max_judge_attempts} to get valid evaluation...")
                llm_messages_history = [{"role": "system", "content": judge_prompts[0]}]

                schema_for_this_call = judge_json_schema_str if llm_judge_id_selected < 10 else None

                response_from_llm = query_llm_agent(llm_input_content, llm_messages_history,
                                                    llm_judge_id_selected, schema_for_this_call)

                if not response_from_llm:
                    print("      LLM Judge returned an empty response.")
                    if attempt_num < max_judge_attempts - 1:
                        continue
                    else:
                        break

                processed_object = postprocessing_response_json(response_from_llm)

                if not isinstance(processed_object, dict):
                    print("      Response not valid JSON or postprocessing failed. Applying Guardrail 1...")
                    response_from_llm = guardrails(judge_prompts[0], llm_input_content, response_from_llm,
                                                   judge_prompts[2], "guardrails_judge_single_json_invalid.txt",
                                                   llm_judge_id_selected, join_folder_files)
                    processed_object = postprocessing_response_json(response_from_llm)
                    if not isinstance(processed_object, dict):
                        print("      Guardrail 1 (JSON structure) failed. Retrying main query if attempts left.")
                        if attempt_num < max_judge_attempts - 1:
                            continue
                        else:
                            break

                        # if we have a dict, normalize and check completeness
                if isinstance(processed_object, dict):
                    normalized_evaluation = normalize_json_object(processed_object)
                    if is_complete_evaluation(normalized_evaluation):
                        judged_json_object = normalized_evaluation
                        print("      LLM Judge provided a valid and complete evaluation.")
                        break
                    else:
                        print("      Evaluation JSON not complete or values incorrect. Applying Guardrail 2...")
                        response_from_llm = guardrails(judge_prompts[0], llm_input_content, response_from_llm,
                                                       judge_prompts[3], "guardrails_judge_single_content_invalid.txt",
                                                       llm_judge_id_selected, join_folder_files)
                        processed_object = postprocessing_response_json(response_from_llm)
                        if isinstance(processed_object, dict):
                            normalized_evaluation = normalize_json_object(processed_object)
                            if is_complete_evaluation(normalized_evaluation):
                                judged_json_object = normalized_evaluation
                                print("      LLM Judge provided valid evaluation after Guardrail 2.")
                                break
                        print("      Guardrail 2 (content) failed to produce a complete evaluation.")
                        if attempt_num == max_judge_attempts - 1:
                            print(
                                f"      Max attempts reached for Paper ID {paper_unique_id}, SKG {current_skg_being_judged}.")

            if judged_json_object:
                justification_output_filename = add_timestamp_and_llm(llm_judge_id_selected) + \
                                                f"{current_skg_being_judged}_paper_{paper_unique_id}_justification.json"
                justification_filepath = os.path.join(join_folder_results("all_justifications"),
                                                      justification_output_filename)
                try:
                    with open(justification_filepath, 'w', encoding='utf-8') as f_just:
                        json.dump({"justification": judged_json_object.get("justification", "N/A")}, f_just, indent=4)
                    print(f"    Justification saved to: {justification_filepath}")
                except Exception as e_just_save:
                    print(f"    Error saving justification file: {e_just_save}")

                score_key_for_output = f"{paper_unique_id}_{current_skg_being_judged}"
                for criterion_name_iter, score_value in judged_json_object.items():
                    if criterion_name_iter != "justification":
                        if criterion_name_iter in final_judge_evaluation_output[current_skg_being_judged] and \
                                score_key_for_output in final_judge_evaluation_output[current_skg_being_judged][
                            criterion_name_iter]:
                            final_judge_evaluation_output[current_skg_being_judged][criterion_name_iter][
                                score_key_for_output] = score_value
            else:
                print(
                    f"    Failed to obtain valid evaluation for Paper ID {paper_unique_id}, SKG {current_skg_being_judged}.")

    for skg_map_idx, combined_ttl_file_content in enumerate(list_of_all_combined_ttls):
        current_skg_being_judged = skg_iteration_map[skg_map_idx]
        print(f"\n--- LLM Judge evaluating COMBINED TTL for: {current_skg_being_judged.upper()} ---")

        if not combined_ttl_file_content:
            print(f"    Skipping combined TTL for {current_skg_being_judged} (no content).")
            continue

        llm_input_content_combined = combined_ttl_file_content

        judged_json_object_combined = None
        response_from_llm_combined = ""
        max_judge_attempts_combined = 5

        for attempt_num_comb in range(max_judge_attempts_combined):
            print(
                f"    Attempt {attempt_num_comb + 1}/{max_judge_attempts_combined} to get valid evaluation for combined TTL...")
            llm_messages_history_comb = [{"role": "system", "content": judge_prompts[1]}]

            schema_for_this_call_comb = judge_json_schema_str if llm_judge_id_selected < 10 else None
            response_from_llm_combined = query_llm_agent(llm_input_content_combined, llm_messages_history_comb,
                                                         llm_judge_id_selected, schema_for_this_call_comb)

            if not response_from_llm_combined:
                print("      LLM Judge returned an empty response for combined TTL.")
                if attempt_num_comb < max_judge_attempts_combined - 1:
                    continue
                else:
                    break

            processed_object_comb = postprocessing_response_json(response_from_llm_combined)

            if not isinstance(processed_object_comb, dict):
                print("      Combined TTL response not valid JSON. Applying Guardrail 1...")
                response_from_llm_combined = guardrails(judge_prompts[1], llm_input_content_combined,
                                                        response_from_llm_combined,
                                                        judge_prompts[2], "guardrails_judge_combined_json_invalid.txt",
                                                        llm_judge_id_selected, join_folder_files)
                processed_object_comb = postprocessing_response_json(response_from_llm_combined)
                if not isinstance(processed_object_comb, dict):
                    print("      Guardrail 1 (JSON structure) for combined TTL failed.")
                    if attempt_num_comb < max_judge_attempts_combined - 1:
                        continue
                    else:
                        break

            if isinstance(processed_object_comb, dict):
                normalized_evaluation_comb = normalize_json_object(processed_object_comb)
                if is_complete_evaluation(normalized_evaluation_comb):
                    judged_json_object_combined = normalized_evaluation_comb
                    print("      LLM Judge provided valid and complete evaluation for combined TTL.")
                    break
                else:
                    print("      Combined TTL evaluation JSON not complete. Applying Guardrail 2...")
                    response_from_llm_combined = guardrails(judge_prompts[1], llm_input_content_combined,
                                                            response_from_llm_combined,
                                                            judge_prompts[3],
                                                            "guardrails_judge_combined_content_invalid.txt",
                                                            llm_judge_id_selected, join_folder_files)
                    processed_object_comb = postprocessing_response_json(response_from_llm_combined)
                    if isinstance(processed_object_comb, dict):
                        normalized_evaluation_comb = normalize_json_object(processed_object_comb)
                        if is_complete_evaluation(normalized_evaluation_comb):
                            judged_json_object_combined = normalized_evaluation_comb
                            print("      LLM Judge provided valid evaluation for combined TTL after Guardrail 2.")
                            break
                    print("      Guardrail 2 (content) for combined TTL failed.")
                    if attempt_num_comb == max_judge_attempts_combined - 1:
                        print(f"      Max attempts reached for combined TTL, SKG {current_skg_being_judged}.")

        if judged_json_object_combined:
            justification_output_filename_comb = add_timestamp_and_llm(llm_judge_id_selected) + \
                                                 f"{current_skg_being_judged}_combined_justification.json"
            justification_filepath_comb = os.path.join(join_folder_results("all_justifications"),
                                                       justification_output_filename_comb)
            try:
                with open(justification_filepath_comb, 'w', encoding='utf-8') as f_just_comb:
                    json.dump({"justification": judged_json_object_combined.get("justification", "N/A")}, f_just_comb,
                              indent=4)
                print(f"    Justification for combined TTL saved to: {justification_filepath_comb}")
            except Exception as e_just_save_comb:
                print(f"    Error saving combined TTL justification file: {e_just_save_comb}")

            score_key_for_combined_output = f"11_{current_skg_being_judged}"
            for criterion_name_iter_comb, score_value_comb in judged_json_object_combined.items():
                if criterion_name_iter_comb != "justification":
                    if criterion_name_iter_comb in final_judge_evaluation_output[current_skg_being_judged] and \
                            score_key_for_combined_output in final_judge_evaluation_output[current_skg_being_judged][
                        criterion_name_iter_comb]:
                        final_judge_evaluation_output[current_skg_being_judged][criterion_name_iter_comb][
                            score_key_for_combined_output] = score_value_comb
        else:
            print(f"    Failed to obtain valid evaluation for COMBINED TTL, SKG {current_skg_being_judged}.")

    save_final_evaluation_json(final_judge_evaluation_output, llm_judge_id_selected)
    print("\n--- LLM Judge evaluation process completed. ---")


if __name__ == "__main__":
    main()
