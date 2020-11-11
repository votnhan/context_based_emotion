import logging
import logging.config
import os
from pathlib import Path
from utils import read_json


def setup_logging(log_dir, config_file='logger/logger_config.json', 
                    default_level=logging.INFO):
    
    log_dir = Path(log_dir)
    if os.path.exists(config_file):
        cfg_dict = read_json(config_file)
        for _, handler in cfg_dict['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(log_dir / handler['filename'])

        logging.config.dictConfig(cfg_dict)
    else:
        print('Config file for logging is not found !')
        logging.basicConfig(level=default_level)
