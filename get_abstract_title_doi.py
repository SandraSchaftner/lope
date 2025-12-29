"""
ORKG Publication Scraper: Title, DOI, and Abstract Extraction
=============================================================

This script automates the retrieval of metadata (Title, DOI) and abstracts for scientific
publications listed in the Open Research Knowledge Graph (ORKG).

It strictly follows a specific logic required for dataset reproduction:
1.  **ORKG Navigation**: Extracts the Title and DOI.
2.  **DOI Resolution**: Validates DOIs and follows CrossRef redirects.
3.  **Abstract Extraction**: Scrapes abstracts (ScienceDirect, Springer, IEEE, etc.).
4.  **Strict Error Handling**: The script is designed to STOP and save immediately if an
    abstract cannot be found, EXCEPT for one specific known edge-case ID (R143763).

Prerequisites:
- Google Chrome installed.
- Active Desktop Session (Required for `pyautogui` mouse movements).
- Python packages: `selenium`, `undetected-chromedriver`, `pyautogui`, `beautifulsoup4`, `requests`, `pandas`, `python-dotenv`.

Input:
- 'orkg_properties_llm_dimensions_dataset_test.csv'

Output:
- 'publications.json'
"""

import json
import random
import time
import os
import sys
from typing import List, Dict, Union, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Selenium & Driver Imports
import undetected_chromedriver as uc
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, NoSuchElementException
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Human Input Simulation
import pyautogui

# --- CONFIGURATION ---
INPUT_CSV_FILE = "orkg_properties_llm_dimensions_dataset_test.csv"
OUTPUT_JSON_FILE = "publications.json"


def install_webdriver() -> WebDriver:
    """
    Attempts to install and initialize the Chrome WebDriver with specific options
    to avoid bot detection.

    Returns:
        WebDriver: An instance of the Chrome WebDriver if successful.

    Raises:
        Exception: If the WebDriver fails to initialize after 5 attempts.
    """
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")  # Options against bot detection
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-infobars")
    options.add_argument("--incognito")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/132.0.6834.110/111 Safari/537.36")

    for attempt in range(5):
        try:
            driver = uc.Chrome(options=options, headless=False)

            # Overwrite webdriver properties to hide automation
            driver.execute_script("""
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
            """)

            actions = ActionChains(driver)
            actions.move_by_offset(random.randint(100, 500), random.randint(100, 500)).perform()
            return driver
        except Exception as e:
            if attempt < 4:
                time.sleep(10)
            else:
                raise Exception("Failed to initialize WebDriver after 5 attempts.") from e


def get_title(driver: WebDriver) -> Union[str, None]:
    """
    Retries fetching the title until its length is greater than 10 or max 10 attempts are reached.

    Args:
        driver: The Selenium WebDriver instance.

    Returns:
        str: The page title if successful, otherwise None.
    """
    for attempt in range(10):
        try:
            title = driver.find_element(By.XPATH, '//title').get_attribute("textContent")
            if len(title) > 10:
                return title
        except Exception:
            pass
        time.sleep(2)
    return None


def get_title_and_doi(publication: Dict[str, Any], driver: WebDriver) -> Union[str, None]:
    """
    Extracts the DOI and title from a given ORKG page.
    Attempts multiple strategies (metadata tags, links).

    Args:
        publication (Dict[str, Any]): Dictionary containing publication data.
        driver (WebDriver): Selenium WebDriver instance.

    Returns:
        Union[str, None]: The extracted DOI if found, otherwise None.
    """
    url = publication["paper_id"]

    try:
        driver.get(url)
        time.sleep(2)
        title = get_title(driver)
        if title:
            title = title[:-7]  # removing " - ORKG"
            # removing " - Resource":
            if title[-11:] == " - Resource":
                title = title[:-11]
            publication["title"] = title
        else:
            publication["title"] = None

        # First attempt: Look for the DOI link in the <small> tag structure
        try:
            doi_element = driver.find_element(By.XPATH, '//small[contains(text(), "DOI:")]/a')
        except NoSuchElementException:
            doi_element = None

        if doi_element:
            doi = doi_element.get_attribute("href")
            if doi is not None:
                print(doi)
                if is_valid_doi(doi):
                    print("doi valid")
                    return doi
                if is_valid_doi_with_selenium(doi, driver):
                    print("doi valid")
                    return doi

        # Second attempt: Look for the DOI link in the <a> tag structure
        try:
            visit_paper_link = driver.find_element(By.XPATH,
                                                   '//a[@class="dropdown-item" and contains(text(), "Visit paper")]')
        except NoSuchElementException:
            visit_paper_link = None

        if visit_paper_link:
            doi = visit_paper_link.get_attribute("href")
            if doi is not None:
                if is_valid_doi(doi):
                    return doi

        return None

    except Exception as e:
        publication["title"] = None
        return None


def get_doi_from_url(url: str, driver: WebDriver) -> Union[str, None]:
    """
    Helper function to extract DOI from a direct URL (used in testing).

    Args:
        url (str): The URL to scrape.
        driver (WebDriver): Selenium WebDriver instance.

    Returns:
        Union[str, None]: The DOI if found, otherwise None.
    """
    try:
        driver.get(url)
        time.sleep(2)
        title = get_title(driver)
        if title:
            title = title[:-7]
            if title[-11:] == " - Resource":
                title = title[:-11]
            print(title)

        # First attempt
        try:
            doi_element = driver.find_element(By.XPATH, '//small[contains(text(), "DOI:")]/a')
        except NoSuchElementException:
            doi_element = None
        if doi_element:
            doi = doi_element.get_attribute("href")
            if doi is not None:
                print(doi)
                if is_valid_doi(doi):
                    print("doi valid")
                    return doi
                if is_valid_doi_with_selenium(doi, driver):
                    print("doi valid")
                    return doi

        # Second attempt
        try:
            visit_paper_link = driver.find_element(By.XPATH,
                                                   '//a[@class="dropdown-item" and contains(text(), "Visit paper")]')
        except NoSuchElementException:
            visit_paper_link = None
        if visit_paper_link:
            doi = visit_paper_link.get_attribute("href")
            if doi is not None:
                print(doi)
                if is_valid_doi(doi):
                    print("visit paper link valid")
                    return doi

        return None

    except Exception as e:
        return None


def get_crossref_doi(doi: str, driver: WebDriver) -> Union[str, None]:
    """
    Retrieves the actual DOI URL from a CrossRef DOI redirection page.
    """
    try:
        driver.get(doi)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        url = driver.find_element(By.XPATH, '//div[@class="resource-line"]/a').get_attribute("href")
        return url
    except Exception as e:
        return doi


def is_redirected_to_crossref(url: str) -> bool:
    """
    Checks whether a given URL redirects to CrossRef's DOI resolver.
    """
    try:
        response = requests.get(url, allow_redirects=True)
        final_url = response.url
        if "chooser.crossref.org" in final_url:
            return True
        else:
            return False
    except Exception as e:
        return False


def scrolling(driver: WebDriver) -> None:
    """
    Scrolls the webpage in a human-like manner to reduce the likelihood of detection.
    """
    for _ in range(random.randint(1, 3)):
        scroll_by = random.randint(200, 600)
        driver.execute_script(f"window.scrollBy(0, {scroll_by});")
        time.sleep(random.uniform(0.5, 1.5))


def move_mouse(driver: WebDriver) -> None:
    """
    Moves the mouse cursor in a human-like manner using pyautogui.
    """
    try:
        width, height = driver.execute_script("return [window.innerWidth, window.innerHeight];")
        x = random.randint(0, width)
        y = random.randint(0, height)
        pyautogui.moveTo(x, y, duration=random.uniform(1, 3))
    except Exception:
        pass


def get_abstract(doi: str, driver: WebDriver) -> Union[str, None]:
    """
    Retrieves the abstract of a research paper from its DOI URL.
    Handles ScienceDirect bot detection and various publisher layouts.

    Args:
        doi (str): The DOI URL.
        driver (WebDriver): Selenium WebDriver instance.

    Returns:
        Union[str, None]: The extracted abstract text or None.
    """
    if is_redirected_to_crossref(doi):
        doi = get_crossref_doi(doi, driver)
    try:
        driver.get(doi)
        # ScienceDirect bot behavior mitigation
        if "10.1016" in doi:
            time.sleep(random.uniform(1, 3))
            scrolling(driver)
            move_mouse(driver)

        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(random.uniform(2, 3))
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extensive list of selectors for various publishers
        abstract_divs = soup.select('[class*="abstract"], [id*="abstract"], [class*="Abs"], [id*="Abs"], '
                                    '[class*="rich-text"], [class*="rich-input-content"], '
                                    '[class*="book-description"], [id*="divARTICLECONTENTTop"], '
                                    '[class="capsule__text"], [class="article-section__content.en.main"], '
                                    '[class="so-layout-section-body"], [class="c-article-section__content"]')

        if abstract_divs:
            no_duplicates = set(div.get_text(strip=True) for div in abstract_divs)
            no_duplicates = [text for text in no_duplicates if len(text) >= 300]
            no_duplicates = [text for text in no_duplicates if not (
                    text.startswith("Download this article") or text.endswith("DownloadPDF") or
                    text.startswith("Contributors")
            )]
            abstract_text = "\n".join(no_duplicates)

            # Cleaning phrases
            if "Keywords:" in abstract_text and not abstract_text.startswith("Articles|Volume"):
                abstract_text = abstract_text.split("Keywords:")[0]
            if "Abstract" in abstract_text:
                abstract_text = abstract_text.split("Abstract")[1]

            cleanup_phrases = [
                "You do not have access", "Articles|Volume",
                "Research Article|", "Published in:", "ACS PublicationsCopyright",
                "Full ArticleFigures", "ArticleCASGoogle", "ArticleGoogle Scholar",
                "Authors and Affiliations", "REFERENCES1", "Cite this article",
                "ArticlePubMedCASPubMed", "Graphical abstract", "Google Scholar", "Cite this paper",
                "Competing Interest Statement", "This publication is licensed"
            ]
            for phrase in cleanup_phrases:
                if phrase in abstract_text:
                    abstract_text = abstract_text.split(phrase)[0]
            abstract_text = abstract_text[:3500]
            return abstract_text

        return None
    except Exception as e:
        return None


def is_valid_doi(doi: str) -> bool:
    """Checks if a DOI is valid via HTTP request."""
    if doi == "https://doi.org/undefined":
        return False
    try:
        response = requests.get(doi, timeout=10)
        return response.status_code == 200
    except (requests.RequestException, Exception):
        return False


def is_valid_doi_with_selenium(doi: str, driver: WebDriver) -> bool:
    """Checks if a DOI is valid via Selenium (checks for page title)."""
    if doi == "https://doi.org/undefined":
        return False
    try:
        driver.get(doi)
        time.sleep(3)
        page_title = driver.title
        return bool(page_title)
    except WebDriverException as e:
        return False


def store_csv_data(table: dict) -> List[dict]:
    """Converts CSV table data to a list of publication dictionaries."""
    research_problems = table["research_problem"]
    orkg_properties_all = table["orkg_properties"]
    nechakhin_result = table["gpt_dimensions"]
    nechakhin_mappings = table["mappings"]
    nechakhin_alignments = table["alignments"]
    nechakhin_deviations = table["deviations"]
    paper_id = table["paper_id"]
    publications = list()
    for i in range(len(research_problems)):
        publication = {"paper_id": paper_id[i], "research_problem": research_problems[i],
                       "orkg_properties": orkg_properties_all[i],
                       "nechakhin_result": nechakhin_result[i], "nechakhin_mappings": nechakhin_mappings[i],
                       "nechakhin_alignment": nechakhin_alignments[i], "nechakhin_deviation": nechakhin_deviations[i]}
        publications.append(publication)

    return publications


def read_csv(file_name: str) -> dict:
    """Reads a CSV file into a dictionary."""
    df = pd.read_csv(file_name, sep=";")
    table = {column: df[column].tolist() for column in df.columns}
    return table


def main():
    driver = install_webdriver()
    load_dotenv()



    table = read_csv(INPUT_CSV_FILE)
    publications = store_csv_data(table)

    for publication in publications:
        doi = get_title_and_doi(publication, driver)

        # 30 attempts loop for DOI
        i = 0
        while doi is None and i < 30:
            doi = get_title_and_doi(publication, driver)
            i += 1

        abstract = get_abstract(doi, driver)

        # 3 attempts loop for Abstract
        i = 0
        while abstract is None and i < 3:
            abstract = get_abstract(doi, driver)
            i += 1

        publication["abstract"] = abstract

        # If abstract is missing AND it's not the specific ID R143763 -> Save & Quit
        if abstract is None and publication["paper_id"] != "http://orkg.org/orkg/resource/R143763":
            try:
                with open(OUTPUT_JSON_FILE, 'w', encoding="utf-8") as file:
                    json.dump(publications, file, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"Error writing to json: {e}")
            driver.quit()
            return

    # Final Save if loop completes
    try:
        with open(OUTPUT_JSON_FILE, 'w', encoding="utf-8") as file:
            json.dump(publications, file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing to json: {e}")
    driver.quit()


if __name__ == "__main__":
    main()