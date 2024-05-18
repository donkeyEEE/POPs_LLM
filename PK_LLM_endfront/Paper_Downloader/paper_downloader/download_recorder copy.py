import pandas as pd
import os
import json
import datetime

class LiteratureTableManager:
    file_name = "litera_lis.csv"  # 默认的文档名称
    merge_log_name = "merge_log.json"  # 合并日志的文件名称
    table = pd.DataFrame({'Authors':[],'Publication Type':[],'DOI':[],'Article Title':[]}) # 文献清单表格

    def __init__(self, save_path="demo"):
        self.save_path = save_path
        self.file_path = os.path.join(self.save_path, self.file_name)
        self.merge_log_path = os.path.join(self.save_path, self.merge_log_name)
        self.load_create_checkpoint()
        
    def add_literature_from_file(self, file_path):
        """从另一个CSV文件中读取新的文献并添加到表格中，去除重复的Article Title"""
        try:
            new_literature = pd.read_excel(file_path)
        except:
            new_literature = pd.read_csv(file_path)
        new_literature = new_literature[['Authors','Publication Type','DOI','Article Title']]
        
        # 为新加的文献添加is_download列，默认值为0
        new_literature['is_download'] = 0
        # 去除新加文档中和原始文档中重复的Article Title
        new_literature = new_literature[~new_literature['Article Title'].isin(self.table['Article Title'])]
        
        # 记录新加文献数量
        new_literature_count = len(new_literature)
        print(f"新增{new_literature_count}条文献")
        # 合并新的文献数据
        self.table = pd.concat([self.table, new_literature], ignore_index=True)
        self.save_table(self.file_path)
        
        # 记录合并信息
        self.log_merge(file_path, new_literature_count)

    def save_table(self, file_path):
        """保存文献表格到指定路径"""
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        self.table.to_csv(file_path, index=False)

    def log_merge(self, file_path, new_literature_count):
        """记录合并信息到JSON文件"""
        merge_info = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'merged_file': file_path,
            'new_literature_count': new_literature_count
        }
        if os.path.exists(self.merge_log_path):
            with open(self.merge_log_path, 'r') as f:
                merge_log = json.load(f)
        else:
            merge_log = []
        merge_log.append(merge_info)
        with open(self.merge_log_path, 'w') as f:
            json.dump(merge_log, f, indent=4)

    def update_download_status(self, doi, status):
        """根据DOI更新文献的下载状态"""
        if doi in self.table['DOI'].values:
            self.table.loc[self.table['DOI'] == doi, 'is_download'] = status
            self.save_table(self.file_path)
            print(f"文献DOI为 {doi} 的下载状态已更新为 {status}")
        else:
            print(f"文献DOI为 {doi} 的文献不在清单中")
    def get_downloaded_num(self):
        # 获取下载文献数目
        return sum(self.table['is_download'])
    
    def download_new_literature(self):
        from paper_downloader.DOI_Download import download_paper_by_doi
        for _doi in self.table['DOI']:
            _doi = str(_doi)
            print(_doi)
            is_download = download_paper_by_doi(_doi,path=f"{self.save_path}/literature_PDF")
            if is_download:
                self.update_download_status(doi=_doi,status=1)
            else:
                self.update_download_status(doi=_doi,status=0)
                
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
                print(undownloaded_literature[['Authors', 'Article Title', 'DOI']])
            return undownloaded_literature[['Authors', 'Article Title', 'DOI']]
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
        self.save_table(self.file_path)
        
        # 打印操作结果
        print(f"已删除所有未下载的文献，当前文献数目为: {len(self.table)}")

    def load_create_checkpoint(self):
        if os.path.exists(self.save_path) ==False:
            os.mkdir(self.save_path)
        if os.path.exists(f"{self.save_path}/literature_PDF") ==False:
            os.mkdir(f"{self.save_path}/literature_PDF")
        if os.path.exists(self.file_path) ==False:
            self.table.to_csv(self.file_path)
        if os.path.exists(self.merge_log_path) ==False:
            with open(self.merge_log_path, 'w') as f:
                json.dump([], f, indent=4)
        self.read_table(self.file_path)
        self.load_merge_log()
        
    def read_table(self, file_path):
        """从指定路径读取文献表格"""
        if os.path.exists(file_path):
            self.table = pd.read_csv(file_path)
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