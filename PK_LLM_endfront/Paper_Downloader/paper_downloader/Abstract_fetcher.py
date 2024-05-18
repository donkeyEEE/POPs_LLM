import requests
from bs4 import BeautifulSoup
"""
    根据文献的DOI或PMID从PubMed获取其摘要信息。
    
    参数:
    litera_id (str): 文献的DOI或PMID。
    _type (str): 指定litera_id是DOI还是PMID，默认为"DOI"。
    
    返回:
    str: 文献的摘要文本。如果无法获取摘要，则返回错误信息或提示无摘要信息。
    
    异常:
    如果请求过程中发生错误，将捕获异常并返回错误描述。
"""
    
def fetch_abstract(litera_id: str, _type="DOI"):
    if _type == "DOI":
        base_url = "https://pubmed.ncbi.nlm.nih.gov/pubmed/?term="
    elif _type == "PMID":
        base_url = "https://pubmed.ncbi.nlm.nih.gov/"
    
    url = f"{base_url}{litera_id}/"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查响应状态码
        soup = BeautifulSoup(response.text, 'html.parser')
        abstract_div = soup.find('div', class_='abstract-content')
        if abstract_div:
            return abstract_div.get_text(strip=True)
        else:
            return "No abstract available for this PMID."
    except Exception as e:
        return f"No abstract available for this PMID."
