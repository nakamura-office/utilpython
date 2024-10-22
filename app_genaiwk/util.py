
import requests
from typing import List, Dict, Any, Union
from googleapiclient.discovery import build
from bs4 import BeautifulSoup

class WebSearchUtil:
    def __init__(self, api_key: str, cse_id: str) -> None:
        self.api_key = api_key
        self.cse_id = cse_id
        self.service = build("customsearch", "v1", cache_discovery=False, developerKey=self.api_key)


    def web_search(self, keyword: str, url: str = "") -> List[Dict[str, Any]]:
        """指定されたキーワードでWeb検索を行い、必要な情報を返す

        Args:
            keyword: キーワード
            url: 検索対象のURL

        Returns:
            検索結果
        """
        start_index = 1
        if url != "":
            keyword += f" site:{url}"

        response = self.service.cse().list(q=keyword, cx=self.cse_id, num=5, start=start_index).execute()

        if response["searchInformation"]["totalResults"] == "0":
            return []
        else:
            return response["items"]


    def get_web_page(self, search_result: Dict[str, Any], message: str) -> Dict[str, Union[str, None]]:
        """指定されたURLのWebページを取得し、検索ワードに関連する部分を取得する

        Args:
            search_result: google検索結果
            keyword: 検索ワード

        Returns:
            Webページの内容
        """
        try:
            headers = {'User-Agent': 'Mozilla/5.0', "referer": 'https://www.google.com/'}
            r = requests.get(url=search_result.get("link"), headers=headers, timeout=(3.0, 7.5))
            content_type_encoding = r.encoding if r.encoding != 'ISO-8859-1' else None
            soup = BeautifulSoup(r.content, 'html.parser', from_encoding=content_type_encoding)
            input_text = soup.find("body").text
            input_text = input_text.replace("\n", "").replace("\r", "").replace("\t", "").replace(" ", "")
        except Exception as e:
            input_text = "Webページの取得に失敗しました。"

        result = {}
        result['title'] = search_result.get("title")
        result['link'] = search_result.get("link")
        result['abstract'] = input_text

        return result

