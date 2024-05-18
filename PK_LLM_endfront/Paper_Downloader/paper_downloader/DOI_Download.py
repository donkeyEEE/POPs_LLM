import requests
import os
from pathlib import Path
from paper_scraper_main import paperscraper

def download_paper_by_doi(doi:str,path='literature_PDF',name = None)->bool:
    papers = paperscraper.search_papers(doi,
                                    limit=10,
                                    pdir=path,
                                    search_type='doi'
                                    )
    if len(papers) >0 :
        if name is not None:
            ori_path = Path(list(papers.keys())[0])
            new_pdf_path = ori_path.with_name(name)
            os.rename(ori_path,new_pdf_path)
        print(f'下载DOI为{doi}的论文成功。')
        return True
    else:
        return False



def _download_paper_by_doi(doi,path='literature_PDF',name=None):
    """
    根据DOI下载论文。
    参数：
    doi: 论文的数字对象标识符（DOI）。
    如果找到下载链接，将尝试下载论文。
    """
    base_url = 'https://sci-hub.st/'
    full_url = base_url + doi
    try:
        response = requests.get(full_url)
        if response.status_code == 200:
            download_url = extract_download_url(response.text)
            if download_url:
                if save_paper_from_url(download_url,path,file_name=name):
                    print(f'下载DOI为{doi}的论文成功。')
                    return True
                else:
                    download_url = f"https://pubs.acs.org/doi/epdf/{doi}?download=True"
                    if save_paper_from_url(download_url,path,file_name=name):
                        print(f'下载DOI为{doi}的论文成功。')
                        return True
                    else:
                        print('请求失败。')
                        return False
            else:
                print('未找到下载链接。')
                return False
        else:
            print('请求失败。')
            return False
    except Exception as e:
        print(f'处理过程中出现错误: {e}')

def extract_download_url(html_content):
    """
    从HTML内容中提取论文的下载链接。
    参数：
    html_content: 从网页获取的HTML内容。
    返回下载链接，如果未找到则返回None。
    """
    for line in html_content.split('\n'):
        if 'location.href=' in line:
            start = line.find('location.href=') + len('location.href=')
            end = line.find('?download=true')
            url_part = line[start:end].strip("'")
            if url_part.startswith('//'):
                return 'https:' + url_part
            return 'https://sci-hub.st' + url_part
    return None

def save_paper_from_url(url,path = 'literature_PDF',file_name = None):
    """
    从提供的URL下载并保存论文。
    参数：
    url: 论文下载链接
    下载成功后会保存到本地文件。
    """
    if file_name is None:
        file_name = url.split('/')[-1].split('?')[0]
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(f'{path}/{file_name}', 'wb') as file:
                file.write(response.content)
                print(f'文件 {file_name} 下载成功！')
                return True
        else:
            print('下载失败。')
            return False
    except Exception as e:
        print(f'下载过程中出现错误: {e}')
        return False


