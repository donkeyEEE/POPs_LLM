import sys
from datetime import datetime

from loguru import logger as _logger
import os
import os.path
from pathlib import Path
# 获取当前脚本的路径
def p():
    # 当前程序的路径
    current_file_path = __file__
    # 获取此程序的父目录的父目录
    grandparent_directory = os.path.dirname(os.path.dirname(current_file_path))
    return Path(grandparent_directory)

def define_log_level(print_level="INFO", logfile_level="DEBUG"):
    """Adjust the log level to above level"""
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d")
    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(p() / f"logs/{formatted_date}.txt", level=logfile_level)
    return _logger


logger = define_log_level()