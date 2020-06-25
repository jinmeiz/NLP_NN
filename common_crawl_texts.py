import json
import requests
import io
import gzip
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request


DATE_COMMONCRAWL = '2014-10'


def request_url(url_name, date_label: str = None):
    """
    Obtain the response for a url
    """

    if not date_label:
        date_label = '2013-20'

    if not url_name:
        return None

    resp = requests.get(f'http://index.commoncrawl.org/CC-MAIN-{date_label}-index?url={url_name}&output=json')

    pages = None
    try:
        pages = [json.loads(x) for x in resp.text.strip().split('\n')]
    except:
        # print('Exception/Error from loading response:', resp.text[:15])
        # print('Problematic url:', url_name)
        return None

    # multiple pages may have been found - find the first valid page
    for page in pages:
        # return the first valid page
        if page.get('status') == '200':
            return page

    return None


def get_url_reponse(url_name: str, date_label: str = None):
    page = request_url(url_name, date_label)

    if not page:
        return None

    # calculate the start and the end of the relevant byte range
    # (each WARC file is composed of many small GZIP files stuck together)
    offset, length = int(page['offset']), int(page['length'])
    offset_end = offset + length - 1

    # get the file via HTTPS so we don't need to worry about S3 credentials
    # getting the file on S3 is equivalent however - you can request a Range
    prefix = 'https://commoncrawl.s3.amazonaws.com/'
    # We can then use the Range header to ask for just this set of bytes
    resp = requests.get(prefix + page['filename'], headers={'Range': 'bytes={}-{}'.format(offset, offset_end)})

    # page is stored compressed (gzip) to save space
    # extract it using the GZIP library
    raw_data = io.BytesIO(resp.content)
    f = gzip.GzipFile(fileobj=raw_data)

    # What we have now is just the WARC response, formatted:
    data = f.read()
    warc_header_response = None
    try:
        warc_header_response = data.decode("utf-8").strip().split('\r\n\r\n', 2)
    except:
        # print('Exception/Error from decoding warc with url', url_name)
        pass

    if not warc_header_response:
        return None

    if len(warc_header_response) < 3:
        return None

#     header = warc_header_response[1]
#     http_status = header.split()[1]
#     if http_status != '200':
#         return None

    return warc_header_response[-1]


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    """
    Extract text from html
    """
    
    if not body:
        return None

    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)


def extract_url_text(url_name, date_label: str = None):
    """
    Given a url, return the webpage text
    """

    # get response from common crawl
    reponse = get_url_reponse(url_name, date_label)

    # parse html from reponse
    text = text_from_html(reponse)
    return text
