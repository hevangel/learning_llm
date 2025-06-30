import os
from dotenv import load_dotenv, find_dotenv                                                                                                                                   
def load_env():
    _ = load_dotenv(find_dotenv())

def get_ai21_api_key():
    load_env()
    ai21_api_key = os.getenv("AI21_API_KEY")
    return ai21_api_key

import json, requests, re
from bs4 import BeautifulSoup

def get_full_filing_text(ticker: str) -> dict:
    headers = {'User-Agent': 'Company Name CompanyEmail@domain.com'}

    try:
        # Step 1: Get the CIK number
        cik_lookup_url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&owner=exclude&action=getcompany"
        response = requests.get(cik_lookup_url, headers=headers)
        response.raise_for_status()
        cik_match = re.search(r'CIK=(\d{10})', response.text)
        if not cik_match:
            return {"error": f"CIK not found for ticker {ticker}"}
        cik = cik_match.group(1)
        # Step 2: Get the latest 10-Q filing
        filing_lookup_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-Q&dateb=&owner=exclude&count=1"
        response = requests.get(filing_lookup_url, headers=headers)
        response.raise_for_status()
        doc_link_match = re.search(r'<a href="(/Archives/edgar/data/[^"]+)"', response.text)
        if not doc_link_match:
            return {"error": f"Latest 10-Q filing not found for ticker {ticker}"}
        doc_link = "https://www.sec.gov" + doc_link_match.group(1)
        # Step 3: Get the index page of the filing
        response = requests.get(doc_link, headers=headers)
        response.raise_for_status()

        # Step 4: Find the link to the actual 10-Q document
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', class_='tableFile')
        if not table:
            return {"error": "Unable to find the document table"}

        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 4 and '10-Q' in cells[3].text:
                doc_href = cells[2].a['href']
                full_doc_url = f"https://www.sec.gov{doc_href}"
                break
        else:
            return {"error": "10-Q document link not found in the index"}

        # Remove 'ix?doc=/' from the URL
        full_doc_url = full_doc_url.replace('ix?doc=/', '')
        # Step 5: Get the actual 10-Q document
        response = requests.get(full_doc_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from the document
        full_text = ' '.join(soup.stripped_strings)
        # limit the full_text to 250000 characters
        full_text = full_text[:250000]

        return {
            "ticker": ticker,
            "filing_type": "10-Q",
            "filing_url": full_doc_url,
            "full_text": full_text,
            "full_text_length": len(full_text)
        }

    except requests.RequestException as e:
        return {"error": f"HTTP request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


def sec_10q(ticker: str) -> dict:
    headers = {'User-Agent': 'Company Name CompanyEmail@domain.com'}

    try:
        # Step 1: Get the CIK number
        cik_lookup_url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&owner=exclude&action=getcompany"
        response = requests.get(cik_lookup_url, headers=headers)
        response.raise_for_status()
        cik_match = re.search(r'CIK=(\d{10})', response.text)
        if not cik_match:
            return {"error": f"CIK not found for ticker {ticker}"}
        cik = cik_match.group(1)
        # Step 2: Get the latest 10-Q filing
        filing_lookup_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-Q&dateb=&owner=exclude&count=1"
        response = requests.get(filing_lookup_url, headers=headers)
        response.raise_for_status()
        doc_link_match = re.search(r'<a href="(/Archives/edgar/data/[^"]+)"', response.text)
        if not doc_link_match:
            return {"error": f"Latest 10-Q filing not found for ticker {ticker}"}
        doc_link = "https://www.sec.gov" + doc_link_match.group(1)
        # Step 3: Get the index page of the filing
        response = requests.get(doc_link, headers=headers)
        response.raise_for_status()

        # Step 4: Find the link to the actual 10-Q document
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', class_='tableFile')
        if not table:
            return {"error": "Unable to find the document table"}

        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 4 and '10-Q' in cells[3].text:
                doc_href = cells[2].a['href']
                full_doc_url = f"https://www.sec.gov{doc_href}"
                break
        else:
            return {"error": "10-Q document link not found in the index"}

        # Remove 'ix?doc=/' from the URL
        full_doc_url = full_doc_url.replace('ix?doc=/', '')
        # Step 5: Get the actual 10-Q document
        response = requests.get(full_doc_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text from the document
        full_text = ' '.join(soup.stripped_strings)
        # limit the full_text to 250000 characters
        full_text = full_text[:250000]

        return {
            "ticker": ticker,
            "filing_type": "10-Q",
            "filing_url": full_doc_url,
            "full_text": full_text,
            "full_text_length": len(full_text)
        }

    except requests.RequestException as e:
        return {"error": f"HTTP request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}

def file_upload(client):
  text = open('Nvidia_10K_20240128.txt', 'r', encoding='utf-8').read()
  new_file_name = 'Nvidia_10K_2024_' + str(uuid.uuid4().hex) + '.txt'

  with open(new_file_name, 'w') as file:
      file.write(text)

  time.sleep(5)

  file_path = './' + new_file_name
  label = '10k_'+ str(uuid.uuid4().hex)

  file_id = client.library.files.create(
      file_path=file_path,
      labels=[label]
  )
  time.sleep(30)
  return file_id

def call_convrag(client, message):
    # Convert chat history to convrag messages format
    DEFAULT_RESPONSE = "I'm sorry, I cannot answer your questions based on the documents I have access to."

    try:
        chat_response = client.beta.conversational_rag.create(
            messages=message,
            query_extraction_model = 'jamba-1.5-large',
            question_answering_model = 'jamba-1.5-large',
            # labels=["10q"],
            # max_segments = 15,
            # retrieval_similarity_threshold = 0.8, # Range: 0.5 – 1.5
            # retrieval_strategy = 'segments',  # ['segments', 'add_neighbors', 'full_doc']
            # max_neighbors = 2, # Used only when retrieval_strategy = 'add_neighbors'
            # hybrid_search_alpha = 0.98 # Range: 0.0 – 1.0. 1.0 means using only dense embeddings; 0.0 means using only keyword search.
        )

    except Exception as e:
        raise Exception(f"Error occurred: {e}")

    if chat_response.context_retrieved and not chat_response.answer_in_context:
      # context_retrieved: [boolean] True if the RAG engine was able to find segments related to the user's query.
      # answer_in_context: [boolean] True if an answer was found in the provided documents.
        response = SimpleNamespace(choices=[SimpleNamespace(content=DEFAULT_RESPONSE)], sources=[SimpleNamespace(text="", file_name="")])
    else:
        response = chat_response

    return response

