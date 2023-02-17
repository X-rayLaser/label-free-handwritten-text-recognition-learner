import logging.config

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

handler = logging.FileHandler('lafhterlearn.log')
handler.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
handler.setFormatter(formatter)

# add ch to logger
logger.addHandler(handler)
