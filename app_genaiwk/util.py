import datetime

import requests
from typing import List, Dict, Any, Union
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, Image, Content
from vertexai.preview import generative_models
from google.cloud import storage

# geminiのモデル名
model_gemini_pro10 = "gemini-1.0-pro-002"
model_gemini_pro15 = "gemini-1.5-pro-002"
model_gemini_pro15_flash = "gemini-1.5-flash-002"

# テキストのセーフティ設定は最低にしておく（デモのため）
safety_settings_NONE = {
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
}


class WebSearchUtil:
    def __init__(self, api_key: str, cse_id: str) -> None:
        self.api_key = api_key
        self.cse_id = cse_id
        self.service = build("customsearch", "v1", cache_discovery=False, developerKey=self.api_key)


    def web_search(self, keyword: str, url: str = "", start_index: int = 1, search_num: int = 5) -> List[Dict[str, Any]]:
        """指定されたキーワードでWeb検索を行い、必要な情報を返す

        Args:
            keyword: キーワード
            url: 検索対象のURL
            start_index: 検索開始位置
            search_num: 検索数

        Returns:
            検索結果
        """
        if url != "":
            keyword += f" site:{url}"

        response = self.service.cse().list(q=keyword, cx=self.cse_id, num=search_num, start=start_index).execute()

        if response["searchInformation"]["totalResults"] == "0":
            return []
        else:
            return response["items"]


    def get_web_page(self, search_result: Dict[str, Any]) -> Dict[str, Union[str, None]]:
        """指定されたURLのWebページを取得し、検索ワードに関連する部分を取得する

        Args:
            search_result: google検索結果

        Returns:
            Webページの内容
        """
        try:
            headers = {'User-Agent': 'Mozilla/5.0', "referer": 'https://www.google.com/'}
            r = requests.get(url=search_result.get("link"), headers=headers, timeout=(3.0, 7.5))
            content_type_encoding = r.encoding if r.encoding != 'ISO-8859-1' else None
            soup = BeautifulSoup(r.content, 'html.parser', from_encoding=content_type_encoding)
            html_body = soup.find("body").text
            input_text = soup.find("body").text
            input_text = input_text.replace("\n", "").replace("\r", "").replace("\t", "").replace(" ", "")
        except Exception as e:
            input_text = "Webページの取得に失敗しました。"

        result = {}
        result['title'] = search_result.get("title")
        result['link'] = search_result.get("link")
        result['abstract'] = input_text
        result['html_body'] = html_body

        return result


    def get_search_summary(self, keyword: str, url: str = "", start_index: int = 1, search_num: int = 5) -> str:
        """指定されたキーワードでWeb検索を行い、情報を集約して返す

        Args:
            keyword: キーワード
            url: 検索対象のURL
            start_index: 検索開始位置
            search_num: 検索数

        Returns:
            検索結果の要約
        """

        # Web検索
        search_results = self.web_search(keyword=keyword, url=url, start_index=start_index, search_num=search_num)

        # Webページの取得
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.get_web_page, item) for item in search_results]
        # 結果を収集
        summary_text = ""
        for future in futures:
            func_result = future.result()
            title = func_result.get("title")
            link = func_result.get("link")
            abstract = func_result.get("abstract")
            summary_text += f"Webページのタイトル: {title}\nURL: {link}\n"
            summary_text += f"取得情報：\n{abstract}\n\n"

        return summary_text


class LLMUtil:
    def __init__(self, model_name, system_instruction=None, response_mime_type="text/plain", max_output_tokens=2048):
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.max_output_tokens = max_output_tokens
        self.response_mime_type = response_mime_type
        self.generation_config = {
            "temperature": 0.0,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": self.response_mime_type
            }
        self.model = GenerativeModel(model_name=self.model_name, system_instruction=self.system_instruction)

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=safety_settings_NONE,
                )
        except Exception as e:
            print(f'error={e}')
            return "I'm sorry, I can't answer that."
        return response

    def chat(self, message, chat_history=[]) -> str:
        message_history = []
        for row in chat_history:
            input_from_user = row[0]
            output_from_llm = row[1]
            message_history.append(Content(role="user", parts=[Part.from_text(input_from_user)]))
            message_history.append(Content(role="model", parts=[Part.from_text(output_from_llm)]))
        chat = self.model.start_chat(history=message_history)
        try:
            response = chat.send_message(content=message, safety_settings=safety_settings_NONE, generation_config=self.generation_config)
        except vertexai.generative_models._generative_models.ResponseBlockedError as e:
            print(e)
            return "I'm sorry, I can't answer that."
        return response.text


def check_gc_notification(bucket_name: str, file_name_base: str, threshold: int) -> bool:
    # google storageの初期化
    storage_client = storage.Client()
    # bucketの取得
    bucket = storage_client.bucket(bucket_name)
    # file_name_baseで始まるファイルのリストを取得
    blobs = bucket.list_blobs(prefix=file_name_base)
    # blobsをファイル名の降順でソート
    blobs = sorted(blobs, key=lambda x: x.name, reverse=True)
    # 1件目のファイルの情報を出力
    blob = blobs[0]
    # ファイル名(budget_over_20241115_10.txt)の中のパーセントを取得
    percent = int(blob.name.split("_")[3].split(".")[0])
    
    if percent >= threshold:
        return True
    else:
        return False