"""
유틸리티 함수 모듈
"""
from typing import List, Dict, Any
from datetime import datetime
from logging_utils import info, success, debug


def format_search_results(search_results: List[Dict[str, Any]]) -> str:
    """
    Tavily 검색 결과를 LLM이 읽기 쉬운 형식으로 포맷팅합니다.
    
    Args:
        search_results: Tavily 검색 결과 리스트
        
    Returns:
        str: 포맷팅된 검색 결과 텍스트
    """
    info("검색 결과 포맷팅", kv={"count": str(len(search_results) if search_results else 0)})
    formatted_text = ""
    
    for idx, result in enumerate(search_results, 1):
        formatted_text += f"\n## 검색 결과 {idx}\n"
        formatted_text += f"**출처:** {result.get('url', 'N/A')}\n"
        formatted_text += f"**제목:** {result.get('title', 'N/A')}\n"
        formatted_text += f"**내용:**\n{result.get('content', 'N/A')}\n"
        formatted_text += "\n---\n"
    
    success("포맷팅 완료", kv={"sections": str(len(search_results) if search_results else 0)})
    return formatted_text


def extract_urls(search_results: List[Dict[str, Any]]) -> List[str]:
    """
    검색 결과에서 URL만 추출합니다.
    
    Args:
        search_results: Tavily 검색 결과 리스트
        
    Returns:
        List[str]: URL 리스트
    """
    urls = []
    if search_results:
        for result in search_results:
            if isinstance(result, dict) and "url" in result:
                urls.append(result["url"])
    info("URL 추출", kv={"count": str(len(urls))})
    return urls


def save_report(report: str, topic: str, sources: List[str], output_dir: str = "reports") -> str:
    """
    생성된 리포트를 파일로 저장합니다.
    
    Args:
        report: 리포트 본문
        topic: 리포트 주제
        sources: 참고 URL 리스트
        output_dir: 저장할 디렉토리 경로
        
    Returns:
        str: 저장된 파일 경로
    """
    import os
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 생성 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '_')).strip()
    safe_topic = safe_topic.replace(' ', '_')[:50]  # 최대 50자
    filename = f"{timestamp}_{safe_topic}.md"
    filepath = os.path.join(output_dir, filename)
    
    # 리포트 내용 구성
    content = report
    
    if sources:
        content += "\n\n---\n\n## 📚 참고 문헌\n\n"
        for idx, url in enumerate(sources, 1):
            content += f"{idx}. {url}\n"
    
    # 파일 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filepath


def validate_api_key(api_key: str) -> bool:
    """
    API Key 형식이 유효한지 검증합니다.
    
    Args:
        api_key: 검증할 API Key
        
    Returns:
        bool: 유효 여부
    """
    if not api_key:
        return False
    
    # 기본값이거나 너무 짧으면 유효하지 않음
    if api_key == "your_tavily_api_key_here" or len(api_key) < 10:
        return False
    
    return True
