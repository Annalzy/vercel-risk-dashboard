"""
Vercel Serverless Function — /api
"""

import csv
import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-pro"

SYSTEM_PROMPT = """你是一位资深企业地缘风险分析师，服务于一家跨国半导体企业的战略部门。
你的任务：仅依据下方提供的【全部风险事件数据】来回答用户的问题。

规则：
1. 只使用提供的数据，不要引入外部知识或推测数据中没有的信息。
2. 如果数据不足以支撑结论，请明确说"现有数据不足以判断"并说明缺少什么信息。
3. 回答要专业、简洁、结构化，使用中文。
4. 引用具体事件时注明日期、国家和来源。
5. 如果用户问的是总结/趋势类问题，给出数据驱动的分析而非空泛评论。"""


def load_events():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "risk_events_clean.csv")
    events = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            events.append({
                "编号": row.get("news_id", ""),
                "日期": row.get("发布日期", ""),
                "国家": row.get("国家", ""),
                "风险类型": row.get("风险类型", ""),
                "标题": row.get("标题", ""),
                "摘要": row.get("正文摘要", ""),
                "来源": row.get("来源", ""),
            })
    return events


def build_user_prompt(events, question):
    events_json = json.dumps(events, ensure_ascii=False, indent=2)
    return f"""## 全部风险事件数据（共 {len(events)} 条）
{events_json}

## 用户问题
{question}

请基于以上数据分析回答。"""


def call_deepseek(api_key, question, events):
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(events, question)},
        ],
        "temperature": 0.7,
        "max_tokens": 2000,
    }).encode("utf-8")

    req = Request(DEEPSEEK_URL, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })

    with urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw)
            question = (data.get("question") or "").strip()

            if not question:
                return self._reply(400, {"error": "请输入问题"})

            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                return self._reply(500, {"error": "未配置 DEEPSEEK_API_KEY"})

            answer = call_deepseek(api_key, question, load_events())
            return self._reply(200, {"answer": answer})

        except json.JSONDecodeError:
            self._reply(400, {"error": "请求格式错误"})
        except URLError as e:
            self._reply(502, {"error": f"DeepSeek API 失败: {e}"})
        except Exception as e:
            self._reply(500, {"error": f"服务器错误: {e}"})

    def _reply(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
