#!/usr/bin/env python
# vim: set ft=python :
"""
Given a PubMed ID (PMID), fetch its XML metadata via the eFetch web service, extract the abstract,
then run the abstract through the Glida named entity recognition web service. Write the JSON output
to the file "<PMID>.json".
"""

from __future__ import annotations

import functools
import operator
from pathlib import Path
from typing import Annotated

import lxml.etree
import typer
from httpx import URL, Client, Timeout

# https://www.ncbi.nlm.nih.gov/books/NBK25499/#chapter4.EFetch
EFETCH_URL = URL("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi")

# https://grounding.indra.bio/apidocs
GLIDA_BASE_URL = URL("https://grounding.indra.bio")
GLIDA_ANNOTATE_ENDPOINT = URL("/annotate")


# HTTPX client that
#   * raises on 4xx or 5xx status
#   * connects with HTTP2
#   * times out after 5 seconds (this is the default, but better to make this explicit)
DefaultClient = functools.partial(
    Client,
    event_hooks={"response": [operator.methodcaller("raise_for_status")]},
    http2=True,
    timeout=Timeout(5.0),
)


def fetch_pubmed_xml(pmid):
    """
    Given a PubMed ID, fetch its XML metadata via the eFetch web service. Returns XML (bytes).
    """
    with DefaultClient() as client:
        response = client.get(EFETCH_URL, params={"db": "pubmed", "id": pmid, "retmode": "xml"})
    return response.content


def extract_abstract_from_pubmed_xml(xml):
    """
    Extract the abstract from a PubMed article's XML metadata. Returns str.
    """
    root = lxml.etree.fromstring(xml)
    return "".join(root.xpath("//AbstractText/text()"))


def apply_glida_ner_to_text(text):
    """
    Run the given text through the Glida named entity recognition web service. Returns JSON (bytes).
    """
    with DefaultClient(base_url=GLIDA_BASE_URL) as client:
        response = client.post(GLIDA_ANNOTATE_ENDPOINT, json={"text": text})
    return response.content


def main(pmid: Annotated[str, typer.Argument(help="PubMed article ID")]):
    pubmed_xml = fetch_pubmed_xml(pmid)
    abstract = extract_abstract_from_pubmed_xml(pubmed_xml)
    glida_ner_json = apply_glida_ner_to_text(abstract)

    # Write the annotation results to the file "<PMID>.json".
    Path(f"{pmid}.json").write_bytes(glida_ner_json)


if __name__ == "__main__":
    typer.run(main)
