import re
import json
from json import dumps, loads

def remove_think(response):
    """Loại bỏ phần <think> và trả về chuỗi sạch."""
    if not isinstance(response, str):
        print(f"Lỗi: response không phải chuỗi! Dữ liệu nhận được: {type(response)}")
        return ""
    cleaned_response = re.sub(r'<think>[\s\S]*?</think>', '', response)
    cleaned_response = re.sub(r'```json|```', '', cleaned_response).strip()
    cleaned_response = cleaned_response.replace("\n", "").strip()
    return cleaned_response

def remove_only_think(response):
    """Loại bỏ nội dung trong <think> nhưng giữ nguyên định dạng."""
    return re.sub(r'<think>[\s\S]*?</think>', '', response, flags=re.DOTALL)

def reciprocal_rank_fusion(results: list[list], k=100, top_n=4):
    """Hợp nhất kết quả tìm kiếm bằng Reciprocal Rank Fusion."""
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = doc.to_dict() if hasattr(doc, "to_dict") else doc.__dict__
            doc_str = dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        (loads(doc_str), score)
        for doc_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results[:top_n] if top_n else reranked_results

# Hàm tiện ích để xóa tag
def remove_tags(text):
    """Loại bỏ các tag như <ASK>, READY từ chuỗi."""
    if not isinstance(text, str):
        return text
    cleaned_text = text.split(":")[1].strip()
    return cleaned_text