from IPython.display import Image, display
from graph.builder import pipeline


try:
    img = Image(pipeline.get_graph(xray=True).draw_mermaid_png())
    # 실행 가능한 객체의 그래프를 mermaid 형식의 PNG로 그려서 표시합니다. xray=True는 추가적인 세부 정보를 포함합니다.
    with open("generated_graph.png", "wb") as f:
        f.write(img.data)
except:
    # 이 부분은 추가적인 의존성이 필요하며 선택적으로 실행됩니다.
    pass
