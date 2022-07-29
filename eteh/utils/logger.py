import logging

class FileLogger(object):

    def __init__(self,filename,logname):
        logger = logging.getLogger(logname)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self.logger = logger
        self.handler = handler

    def info(self,msg):        
        self.logger.info(msg)

    def debug(self,msg):
        self.logger.debug(msg)

    def warning(self,msg):
        self.logger.warning(msg)
        
