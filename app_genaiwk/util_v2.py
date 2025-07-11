import datetime
import warnings
import re
import json
from datetime import datetime

import requests
from itertools import islice
from typing import List, Dict, Any, Union
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from concurrent.futures import ThreadPoolExecutor

#import vertexai
#from vertexai.preview.generative_models import GenerativeModel, Part, Image, Content, GenerationConfig
#from vertexai.preview import generative_models
from google import genai
from google.genai import types
from google.cloud import storage
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine

from elasticsearch.exceptions import ElasticsearchWarning
from elasticsearch import Elasticsearch
from google.auth.transport.requests import Request
from google.oauth2 import id_token

# geminiのモデル名
model_gemini_pro25_flash = "gemini-2.5-flash"


# テキストのセーフティ設定は最低にしておく（デモのため）
SAFETY_SETTING_NONE=[
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,  # Block none
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,  # Block none
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,  # Block none
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,  # Block none
    ),
]


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
            body_tag = soup.find("body")
            if body_tag is not None:
                html_body = body_tag.text
                input_text = body_tag.text
                input_text = input_text.replace("\n", "").replace("\r", "").replace("\t", "").replace(" ", "")
            else:
                input_text = "bodyタグがありませんでした。"
                html_body = "bodyタグがありませんでした。"
        except Exception as e:
            input_text = "Webページの取得に失敗しました。"
            html_body = "Webページの取得に失敗しました。"

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


    def search_ddg(self, search_query, max_result_num=7, site=None, debug_mode=False) -> list[dict]:
        """
        DuckDuckGoを使って検索を行う関数
        引数:
        search_query: 検索クエリ
        max_result_num: 最大取得数
        site: 検索対象のサイト
        戻り値:
        検索結果のリスト
        [{"title": "タイトル", "snippet": "スニペット", "link": "リンク"}, ...]
        """

        # [0] サイト指定があれば追加
        if site:
            search_query = f"{search_query} site:{site}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
        }

        # for debug
        if debug_mode:
            print(f'start search_ddg keyword:{search_query}')
        # [1] Web検索を実施
        #res = DDGS(headers=headers).text(search_query, region='jp-jp', safesearch='off', backend="lite")
        res = DDGS(headers=headers).text(search_query, region='jp-jp')

        # [2] 結果のリストを分解して戻す
        return [
            {
                "title": r.get('title', ""),
                "snippet": r.get('body', ""),
                "link": r.get('href', "")
            }
            for r in islice(res, max_result_num)
        ]

    def get_info_from_web(
            self,
            question: str, 
            search_query: str,
            project: str,
            location: str = "us-central1", 
            profile: str = "", 
            max_result_num=7, 
            web_info_list=[], 
            site=None, 
            model_name: str=model_gemini_pro25_flash,
            threshold_relevance: int=7,
            threshold_credibility: int=7,
            debug_mode: bool=False) -> list:
        """
        インターネット検索結果から必要な情報を取得する関数
        引数:
        question: ユーザの質問
        search_query: Web検索のクエリー
        profile: ユーザ情報
        max_result_num: 最大取得数
        戻り値:
        Web検索結果から取得した情報
        [{"title": "タイトル", "link": "リンク", "text": "テキスト"}, ...]
        """
        
        # for debug
        if debug_mode:
            print(f'start get_info_from_web_1:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        # jsonの形式を定義
        response_schema = {
            "type": "object",
            "properties": {
                "relevance": {
                    "type": "string",
                    "description": "質問とWebページの関連度、大きいほど関連が高い、0-10の値"
                },
                "credibility": {
                    "type": "string",
                    "description": "Webページの内容の信憑性、想像だけだと信憑性が低い、公式サイトの情報や実際に検証している場合は信憑性が高い、信憑性が高いほど数値が大きい、0-10の値"
                },
            },
            "required": ["relevance", "credibility"]
        }
        # LLMUtilを使用してチャット機能をテスト
        system_instruction = "日本語で回答してください。"
        llm_util = LLMUtil(project=project, location=location)


        # for debug
        if debug_mode:
            print(f'start get_info_from_web_2:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        # search_queryを元にWeb検索を実施
        if site:
            search_result = self.web_search(search_query, url=site, search_num=max_result_num) # DuckDuckGoが制限にかかってエラーになる場合はGoogle Custom Searchを使用する
        else:
            #search_result = self.search_ddg(search_query, max_result_num=max_result_num)
            search_result = self.web_search(search_query, search_num=max_result_num) # DuckDuckGoが制限にかかってエラーになる場合はGoogle Custom Searchを使用する

        # 検索結果をテキストで返す（暫定）
        response_text = ""
        response_list = []

        # for debug
        if debug_mode:
            print(f'start get_info_from_web_3:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        # Web検索結果毎に質問との関連度を評価し、回答に必要な内容を抜粋する
        visited_list = []
        for item in web_info_list:
            visited_list.append(item["link"])
        for idx, item in enumerate(search_result):
            if item["link"] in visited_list:
                continue
            # Webページの内容を取得
            web_page_body = self.get_web_page(item)["html_body"]
            # 連続した空白を単一の空白に置き換える
            if web_page_body is not None:
                web_page_body = re.sub(r'\s+', ' ', web_page_body)
            else:
                web_page_body = ""
            # 最大文字数で制限
            web_page_body = web_page_body[:10000]

            if profile == "":
                prompt = f"""
                質問に回答するためにWeb検索をしました。
                検索した結果の以下のWebページの内容が質問に対して関連があるかと
                Webページの内容が信憑性があるかを判定してください。
                
                質問：```{question}```

                Webページの内容：```{web_page_body}```

                """
            else:
                prompt = f"""
                質問に回答するためにWeb検索をしました。
                検索した結果の以下のWebページの内容が質問に対して関連があるかと
                Webページの内容が信憑性があるかを判定してください。
                判定は質問者の特性も加味して検討してください。
                
                質問者：```{profile}```
                質問：```{question}```

                Webページの内容：```{web_page_body}```
                """
            # 閾値の指定がない場合は関連度と信頼性の評価はしない（暫定、速度改善のため）
            if threshold_relevance is None or threshold_credibility is None:
                relevance = 10
                credibility = 10
            else:
                #"""
                response = llm_util.generate_response(
                    prompt,
                    model_name=model_name, 
                    system_instruction=system_instruction, 
                    response_mime_type="application/json", 
                    response_schema=response_schema,
                    thinking_budget=0
                )
                try:
                    response_json = json.loads(response.text)
                except Exception as e:
                    continue
                relevance = int(response_json["relevance"])
                credibility = int(response_json["credibility"])
            if debug_mode:
                print(f'relevance={relevance}, credibility={credibility}')
            # 関連度と信憑性が一定値以下の場合はスキップ
            if relevance < threshold_relevance or credibility < threshold_credibility:
                continue
            #"""
            response_dict = {}
            response_dict["title"] = item["title"]
            response_dict["link"] = item["link"]
            response_dict["text"] = web_page_body
            response_list.append(response_dict)

        # for debug
        if debug_mode:
            print(f'end get_info_from_web:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        
        return response_list


    def create_web_search_query(self, question: str, 
                                project: str, location: str="us-central1", 
                                profile: str = "", max_query_num: int=5, lang:str="jp", model_name: str=model_gemini_pro25_flash) -> str: 
        """
        ユーザの質問に対してWeb検索を行うためのクエリーを生成する関数
        引数:
        search_query: ユーザの質問
        profile: ユーザ情報
        max_query_num: 生成するクエリーの数
        戻り値:
        Web検索を行うためのクエリー
        {"search_queries":[{"search_query":"クエリー1"},{"search_query":"クエリー2"}]}
        """
        # jsonの形式を定義
        response_schema = {
            "type": "object",
            "properties": {
                "search_queries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "Web検索に利用するクエリー"
                            },
                        },
                        "required": ["search_query"]
                    }
                }
            },
            "required": ["search_queries"]
        }
        # LLMUtilを使用してチャット機能をテスト
        if lang == "jp":
            system_instruction = "日本語で回答してください。"
        elif lang == "en":
            system_instruction = "Please answer in English."
        else:
            system_instruction = "日本語で回答してください。"
        llm_util = LLMUtil(project=project, location=location)

    #        質問にバージョン番号の指定があればそれを含めてください。指定されていない場合はバージョン番号は含めないようにしてください。
        if profile == "":
            prompt = f"""
            以下の質問に答えるためにWeb検索を行います。検索クエリーを{max_query_num}個程度考えてください。
            検索クエリーには個人情報や企業情報などを含めないようにしてください。

            質問：```{question}```
            """
        else:
            prompt = f"""
            以下の質問に答えるためにWeb検索を行います。検索クエリーを{max_query_num}個程度考えてください。
            質問者の特性も加味して検討してください。
            検索クエリーには個人情報や企業情報などを含めないようにしてください。

            質問：```{question}```

            質問者：```{profile}```
            """

        response = llm_util.generate_response(
            prompt,
            model_name=model_name, 
            system_instruction=system_instruction, 
            response_mime_type="application/json", 
            response_schema=response_schema,
            thinking_budget=0
        )
        return response.text


class LLMUtil:
    def __init__(self, project: str, location: str = "us-central1",):
        #self.model = GenerativeModel(model_name=self.model_name, system_instruction=self.system_instruction)
        self.client = genai.Client(vertexai=True, project=project, location=location)

    def generate_response(
            self, 
            prompt, 
            model_name=model_gemini_pro25_flash, 
            max_output_tokens=None, 
            system_instruction=None, 
            response_mime_type="text/plain", 
            temperature=0.0, 
            response_schema=None, 
            thinking_budget=8000,
        ) -> Any:
        # configの設定
        gen_config_kwargs = {
            "temperature": temperature,
            "response_mime_type": response_mime_type,
            "safety_settings": SAFETY_SETTING_NONE,
            "system_instruction":system_instruction
        }
        if max_output_tokens is not None and max_output_tokens > 0:
            gen_config_kwargs["max_output_tokens"] = max_output_tokens
        if response_schema is not None:
            gen_config_kwargs["response_schema"] = response_schema
        generate_content_config = types.GenerateContentConfig(
            thinking_config = types.ThinkingConfig(
                thinking_budget=thinking_budget,
            ),
            **gen_config_kwargs
        )
        #self.generation_config = GenerationConfig(**gen_config_kwargs)
        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=generate_content_config,
                )
        except Exception as e:
            print(f'error={e}')
            return None
        return response

    def chat(self, 
            message, 
            chat_history=[], 
            type="messages",
            model_name=model_gemini_pro25_flash, 
            system_instruction=None, 
            response_mime_type="text/plain", 
            temperature=0.0, 
            max_output_tokens=None, 
            response_schema=None, 
            thinking_budget=8000,
            ) -> str:
        """
        type: "tuple" or "messages"
        """
        # configの設定
        gen_config_kwargs = {
            "temperature": temperature,
            "response_mime_type": response_mime_type,
            "safety_settings": SAFETY_SETTING_NONE,
            "system_instruction":system_instruction
        }
        if max_output_tokens is not None and max_output_tokens > 0:
            gen_config_kwargs["max_output_tokens"] = max_output_tokens
        if response_schema is not None:
            gen_config_kwargs["response_schema"] = response_schema
        generate_content_config = types.GenerateContentConfig(
            thinking_config = types.ThinkingConfig(
                thinking_budget=thinking_budget,
            ),
            **gen_config_kwargs
        )
        message_history = []
        if type == "tuple":
            for user_msg, model_msg in chat_history:
                message_history.append(types.Content(role="user", parts=[types.Part.from_text(user_msg)]))
                message_history.append(types.Content(role="model", parts=[types.Part.from_text(model_msg)]))
        elif type == "messages":
            for row in chat_history:
                if row["role"] == "user":
                    message_history.append(types.Content(role="user", parts=[types.Part.from_text(row["content"])]))
                elif row["role"] == "assistant":
                    # google-genai では 'assistant' は通常 'model' に対応します
                    message_history.append(types.Content(role="model", parts=[types.Part.from_text(row["content"])]))
        else:
            message_history = []

        message_history.append(types.Content(role="user", parts=[types.Part.from_text(message)]))
        try:
            response = self.client.models.generate_content(
                model=model_name,
                contents=message_history,
                config=generate_content_config,
            )
            return response.text
        except Exception as e:
            print(e)
            return "I'm sorry, I can't answer that."


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


class RetrieverUtil:
    def __init__(self, project_id, data_store_id):
        self.project_id = project_id
        self.data_store_id = data_store_id
        self.location = "global"
        
        # クライアントのオプションを設定
        client_options = (
            ClientOptions(api_endpoint=f"{self.location}-discoveryengine.googleapis.com")
            if self.location != "global"
            else None
        )

        # クライアントを初期化
        self.client = discoveryengine.SearchServiceClient(client_options=client_options)


    def search_document(self, search_query: str) -> List[dict]:
        # for debug
        # param_text = search_query.replace("\n", "").replace("\r", "").replace("\t", "").replace(" ", "")
        # print(f"start proc search_document params={param_text[:40]}")
        # logger.info(f"start search_document params={param_text[:40]}")        


        # サービング構成を指定
        serving_config = self.client.serving_config_path(
            project=self.project_id,
            location=self.location,
            data_store=self.data_store_id,
            serving_config="default_config",
        )



        # コンテンツ検索仕様を指定
        content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
            extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    max_extractive_segment_count=1,
                    max_extractive_answer_count=1,
                    return_extractive_segment_score=True
            ),
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True
            ),
            # 要約の生成をしない
            #summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            #     summary_result_count=5,
            #    include_citations=True,
            #    ignore_adversarial_query=True,
            #    ignore_non_summary_seeking_query=True,
            #),
        )

        # 検索リクエストを作成
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=search_query,
            page_size=10,
            content_search_spec=content_search_spec,
            query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
                condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
            ),
            spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
            ),
        )

        response = self.client.search(request)

        # 検索結果から必要な項目を取得する（title,extractive_segments,link,）
        document_summary_list = []

        # for debug
        #print(f'{search_results=}')

        # 検索結果からファイルのCloud Storageリンクを取得
        for result in response.results:
            document_summary = {}
            document_summary["title"] = result.document.derived_struct_data.get("title")
            document_summary["link"] = result.document.derived_struct_data.get("link")
            for ex_seg in result.document.derived_struct_data.get("extractive_segments"):
                document_summary["extractive_segment_page_number"] = ex_seg.get("pageNumber")
                document_summary["extractive_segment_content"] = ex_seg.get("content")
                document_summary["extractive_segment_relevanceScore"] = ex_seg.get("relevanceScore")
                document_summary_list.append(document_summary)

        return document_summary_list


class ElasticSearchUtil:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint       

    def search_document(self, index_name:str, query: str, eval_column:str, target_column:list, token: str="") -> List[dict]:
        """
        単純な単語検索を行う
        """
        # GoogleCloudの認証トークンを取得
        if token == "":
            if not self.endpoint:
                raise ValueError("endpointが設定されていません")
            token = id_token.fetch_id_token(Request(), self.endpoint) or ""
            if not isinstance(token, str):
                token = str(token)
        
        # Elasticsearchクライアントの作成
        es = Elasticsearch(
            [self.endpoint],
            headers={"Authorization": f"Bearer {token}"}  # 認証トークンを指定
        )
        # for debug
        # Elasticsearchの警告を無視
        warnings.filterwarnings('ignore', category=ElasticsearchWarning)

        # キーワード検索
        search_body = {'_source': target_column, 'query': {'match': {eval_column: query}}}
        res = es.search(index=index_name, body=search_body)
        hits = [dict(list(doc['_source'].items()) + [('score', doc['_score'])]) for doc in res['hits']['hits']]
        
        # 内部接続を閉じる
        es.close()

        return hits
