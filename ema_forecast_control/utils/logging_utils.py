import logging
import os

def configure_logging(path: str, verbose: str) -> logging.Logger:    
    logger = logging.getLogger('root')
    logger.handlers = []
    logger.setLevel(logging.INFO)
    if verbose=='log':
        hdlr = logging.FileHandler(os.path.join(path, 'log.txt'))
    elif verbose=='print':
        hdlr = logging.StreamHandler()
    else:
        hdlr = logging.NullHandler()
    hdlr.setLevel(logging.INFO) 
    formatter = logging.Formatter('%(asctime)s: %(name)s: %(levelname)s: %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger