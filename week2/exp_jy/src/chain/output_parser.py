import re
import json
from langchain_core.messages import AIMessage
from langchain_core.output_parsers.string import StrOutputParser


class JSONParser(StrOutputParser):
    def __init__(self):
        super().__init__()

    def parse(self, ai_message: AIMessage):
        text = super().parse(ai_message)
        return json.loads(text)


class SqlParser(StrOutputParser):
    def __init__(self):
        super().__init__()

    def parse(self, ai_message: AIMessage):
        text = super().parse(ai_message)
        pattern = f'{re.escape("```sql")}(.*?){re.escape("```")}'

        # 정규표현식으로 추출
        match = re.search(pattern, text, re.DOTALL)

        # 결과 출력
        extracted_content = match.group(1).strip()
        return extracted_content
