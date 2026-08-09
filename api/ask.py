"""
Vercel Serverless Function — /api/ask
接收用户问题，读取全部风险事件 CSV，调用 DeepSeek API 以企业风险分析师身份回答。
API key 从环境变量 DEEPSEEK_API_KEY 读取，不出现在前端代码中。
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
    """读取清洗后的 CSV，返回事件列表"""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "risk_events_clean.csv")
    events = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    """构建发送给 DeepSeek 的用户消息"""
    events_json = json.dumps(events, ensure_ascii=False, indent=2)
    return f"""## 全部风险事件数据（共 {len(events)} 条）
{events_json}

## 用户问题
{question}

请基于以上数据分析回答。"""


def call_deepseek(api_key, question, events):
    """调用 DeepSeek Chat API"""
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
        result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]


def json_response(handler, status_code, data):
    """发送 JSON 响应"""
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        """CORS 预检"""
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        try:
            # 读取请求体
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length else b"{}"
            data = json.loads(body)
            question = (data.get("question") or "").strip()

            if not question:
                return json_response(self, 400, {"error": "请输入问题"})

            # 检查 API key
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                return json_response(self, 500, {
                    "error": "服务端未配置 DEEPSEEK_API_KEY 环境变量"
                })

            # 加载事件数据
            events = load_events()

            # 调用 DeepSeek
            answer = call_deepseek(api_key, question, events)

            json_response(self, 200, {"answer": answer})

        except json.JSONDecodeError:
            json_response(self, 400, {"error": "请求格式错误"})
        except URLError as e:
            json_response(self, 502, {"error": f"DeepSeek API 请求失败: {str(e)}"})
        except Exception as e:
            json_response(self, 500, {"error": f"服务器内部错误: {str(e)}"})

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    # 重写以便 OPTIONS 也带 CORS
    def end_headers(self):
        self._cors_headers()
        super().end_headers()
