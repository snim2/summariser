#!/usr/bin/env python

"""
Create summaries of scientific papers.

DEPENDENCIES:

* networkx
* sqlite
* nltk
* sklearn
"""

import glob
import networkx as nx
import os
import os.path
import re
import subprocess
import sqlite3
import tempfile

from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# pylint: disable=W0105


__author__ = 'Sarah Mount <s.mount@wlv.ac.uk>'
__date__ = 'May 2013'


__DBFILE = os.path.expanduser('~') + os.sep + '.summeriser.db'
"""This users database of papers."""


__TEMPDIR = tempfile.mkdtemp()
"""A temporary directory for storing .tiff files etc."""


__RESOLUTION = 600
"""Resolution to use when scanning PDF files. Higher is better."""


__RE_COUNT_PAGES = re.compile(r'/Type\s*/Page([^s]|$)', 
                            re.MULTILINE | re.DOTALL)
"""Reg exp pattern for counting pages in a PDF file."""


def count_pdf_pages(filename):
    """Count how many pages a PDF file has.
    Active state recipe 496837:
    http://code.activestate.com/recipes/496837-count-pdf-pages/
    """
    data = file(filename, 'rb').read()
    return len(__RE_COUNT_PAGES.findall(data))


class DbEntry:
    def __init__(self, text, pdf, tags):
        self.text = text
        self.pdf = pdf
        self.tags = tags
        return


def __conv(pdf, page, outfile):
    """Produce a shell command to convert a page of a PDF file to a TIFF.
    Requires ImageMagick to be installed.
    """
    command = ['convert', '-monochrome', '-density ' + str(__RESOLUTION), 
               pdf + '[' + str(page) + ']', outfile]
    return ' '.join(command)


def __textrank(document):
    """Apply TextRank to a given piece of text.

    Implementation by Josh Bohde: 
    http://joshbohde.com/blog/document-summarization    

    Mihalcea, R., & Tarau, P. (2004). TextRank: Bringing order into
    texts. In Proceedings of EMNLP (Vol. 4, No. 4).
    """
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(document)
 
    bow_matrix = CountVectorizer().fit_transform(sentences)
    normalized = TfidfTransformer().fit_transform(bow_matrix)
 
    similarity_graph = normalized * normalized.T
 
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    return sorted(((scores[i],s) for i,s in enumerate(sentences)),
                  reverse=True)


def __cleanup():
    """Remove all temporary files and tempory directory.
    """
    files = glob.glob(__TEMPDIR + os.sep + '*')
    for fn in files:
        os.remove(fn)
    os.rmdir(__TEMPDIR)
    return


def process_directory(directory):
    """Process all PDF files in a given directory.

    For each file, convert each page to a .tiff, OCR the .tiff and
    create a text version of the full file. Apply TextRank to the
    file, then summarise and add to the summary database.
    """
    tiff_file = __TEMPDIR + os.sep + 'page.tiff'
    text_file = __TEMPDIR + os.sep + 'ocr'
    pdfs = glob.glob(directory + os.sep + '*.pdf')
    texts = {}
    summaries = {}
    for pdf in pdfs:
        pages = count_pdf_pages(pdf)
        pdf_text = ''
        # Convert each page of each PDF in the directory to a .tiff and OCR.
        for page in range(pages):
            print 'Processing page', page, 'of', pdf
            ex = subprocess.call(__conv(pdf, page, tiff_file),
                                 shell=True, stdout=None)
            if ex != 0: continue # Bail out if convert fails.
            with open(os.devnull, 'w') as fnull:
                ex = subprocess.call('tesseract {0} {1}'.format(tiff_file,
                                                                text_file),
                                     shell=True, stdout=fnull, stderr=fnull)
            if ex != 0: continue # Bail out if tesseract fails.
            with file(text_file + '.txt', 'r') as fn:
                pdf_text += fn.read()
        if len(pdf_text) > 0:
            texts[os.path.abspath(pdf)] = pdf_text
        print pdf, 'contains', len(pdf_text), 'characters.'
    # Summarise each OCR'd file with textrank.
    for text in texts:
        tr = __textrank(texts[text])
        # A summary is 10% of the overall text.
        s_len = int(len(tr) * 0.1)
        summary = ' '.join([txt for _, txt in tr][:(s_len + 1)])
        summaries[text] = summary
    for pdf in summaries:
        print pdf, 'SUMMARY:', summaries[pdf]
    return


if __name__ == '__main__':
    import sys
    process_directory(sys.argv[1])
    __cleanup()
