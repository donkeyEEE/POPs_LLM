import pandas as pd
import os
import json
import datetime
from .Abstract_fetcher import fetch_abstract
def ensure_path_exists(func):
    """修饰器：确保文件或文件夹路径存在，如果不存在则创建"""
    def wrapper(self, *args, **kwargs):
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(os.path.join(self.save_path, 'literature_PDF'), exist_ok=True)
        return func(self, *args, **kwargs)
    return wrapper

class LiteratureTableManager:
    file_name = "litera_lis.csv"  # 默认的文档名称
    merge_log_name = "merge_log.json"  # 合并日志的文件名称
    table = pd.DataFrame({'Authors':[],'Publication Year':[],'DOI':[],'Title':[],'Abstract':[]}) # 文献清单表格

    def __init__(self, save_path="demo"):
        self.save_path = save_path
        self.file_path = os.path.join(self.save_path, self.file_name)
        self.merge_log_path = os.path.join(self.save_path, self.merge_log_name)
        self.load_create_checkpoint()
    
    @ensure_path_exists
    def add_literature_from_file(self, file_path):
        new_literature = self.read_literature_file(file_path)
        new_literature['is_download'] = 0
        
        # 生成文献编号，确保编号从上次的最后编号开始
        start_num = self.literature_num
        new_literature.index = [f"SL01-{start_num + i}-J" for i in range(len(new_literature))]
        self.literature_num += len(new_literature)  # 更新文献编号
        
        # 去重并更新文献表
        new_literature = new_literature[~new_literature['Title'].isin(self.table['Title'])]
        self.table = pd.concat([self.table, new_literature]) # , ignore_index=True
        
        new_literature_count = len(new_literature)
        print(f"新增{new_literature_count}条文献")
        
        self.save_table()
        self.log_merge(file_path, new_literature_count)

    @ensure_path_exists
    def save_table(self):
        """保存文献表格到指定路径"""
        self.table.to_csv(self.file_path)
    
    def log_merge(self, file_path, new_literature_count):
        """记录合并信息到JSON文件"""
        merge_info = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'merged_file': file_path,
            'new_literature_count': new_literature_count
        }
        self.merge_log.append(merge_info)
        with open(self.merge_log_path, 'w') as f:
            json.dump(self.merge_log, f, indent=4)
            
    def update_download_status(self, doi, status,abstract):
        """根据DOI更新文献的下载状态"""
        if doi in self.table['DOI'].values:
            self.table.loc[self.table['DOI'] == doi, 'is_download'] = status
            self.table.loc[self.table['DOI'] == doi, 'Abstract'] = abstract
            self.save_table()
            print(f"文献DOI为 {doi} 的下载状态已更新为 {status}")
        else:
            print(f"文献DOI为 {doi} 的文献不在清单中")
    def get_downloaded_num(self):
        # 获取下载文献数目
        return sum(self.table['is_download'])
    
    def download_new_literature(self,max_limit = 2 , max_num = -1):
        from paper_downloader.DOI_Download import download_paper_by_doi ,_download_paper_by_doi
        from tqdm import tqdm
        for index, row in tqdm(self.table.iterrows()):
            num_litera = sum(self.table['is_download']==1)
            if max_num != -1 and num_litera > max_num:
                print(f"已经下载{num_litera}篇文献")
                break
            _doi = row['DOI']
            status = row['is_download'] # 文献下载状态
            if type(_doi) !=str:
                status = max_limit+2
            
            if status ==1  or status > max_limit:
                continue
            print(f"已经下载{num_litera},正在下载{_doi}")
            abstract = fetch_abstract(_doi)
            try: 
                _ = download_paper_by_doi(_doi,path=f"{self.save_path}/literature_PDF",name=f"{index}.pdf")
            except:
                _ = False
            if _:
                self.update_download_status(doi=_doi,status=1,abstract=abstract)
            else:
                if _download_paper_by_doi(_doi,path=f"{self.save_path}/literature_PDF",name=f"{index}.pdf"):
                    self.update_download_status(doi=_doi,status=1,abstract=abstract)
                else:
                    self.update_download_status(doi=_doi,status=status + 2,abstract=abstract)
        """初始状态为0，若成果则为1
        失败一次加2，状态超过最大值则跳过
        """

    def list_undownloaded_literature(self,p=False)->list:
        """输出尚未下载成功的文献列表及数目"""
        # 筛选出未下载的文献
        undownloaded_literature = self.table[self.table['is_download'] == 0]
        # 打印未下载文献的数目
        print(f"尚未下载成功的文献数目为: {len(undownloaded_literature)}")
        
        # 如果有未下载的文献，打印详情
        
        if len(undownloaded_literature) > 0:
            if p:
                print("以下是尚未下载成功的文献列表:")
                print(undownloaded_literature[['Authors', 'Title', 'DOI']])
            return undownloaded_literature[['Authors', 'Title', 'DOI']]
        else:
            if p:
                print("所有文献都已成功下载。")
            return []
        
    def remove_undownloaded_literature(self):
        """删除所有未下载的文献"""
        # 筛选出已下载的文献
        downloaded_literature = self.table[self.table['is_download'] == 1]
        
        # 更新文献表格为只包含已下载的文献
        self.table = downloaded_literature
        
        # 保存更新后的文献表格
        self.save_table()
        
        # 打印操作结果
        print(f"已删除所有未下载的文献，当前文献数目为: {len(self.table)}")


        
    def read_table(self, file_path):
        """从指定路径读取文献表格"""
        if os.path.exists(file_path):
            self.table = pd.read_csv(file_path,index_col=0)
            # 如果表格中没有is_download列，则添加该列，默认值为0
            if 'is_download' not in self.table.columns:
                self.table['is_download'] = 0
        else:
            print(f"{file_path}不存在")
            self.table = pd.DataFrame()
    def load_merge_log(self):
        """加载合并日志"""
        if os.path.exists(self.merge_log_path):
            with open(self.merge_log_path, 'r') as f:
                self.merge_log = json.load(f)
        else:
            self.merge_log = []
    
    def read_literature_file(self, file_path):
        """根据文件类型读取文献数据"""
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)[['Authors', 'Publication Year', 'DOI', 'Title']]
        else:
            try:
                return pd.read_csv(file_path)[['Authors', 'Publication Year', 'DOI', 'Title']]
            except:
                return pd.read_csv(file_path,encoding="ISO-8859-1")

    def load_create_checkpoint(self):
            """加载或创建启动时的检查点"""
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.save_path, 'literature_PDF'), exist_ok=True)
            
            if not os.path.exists(self.file_path):
                self.table.to_csv(self.file_path)
            
            if not os.path.exists(self.merge_log_path):
                with open(self.merge_log_path, 'w') as f:
                    json.dump([], f, indent=4)
            
            self.read_table(self.file_path)
            self.load_merge_log()
            self.literature_num = len(self.table) + 1  # 根据已有文献更新编号起始点