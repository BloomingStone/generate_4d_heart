from generate_4d_heart import generate_4d_dvf
from cyclopts import run

import logging

logging.basicConfig(
    filename='processing_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - [%(levelname)s] %(filename)s:%(lineno)d -- %(message)s',
    encoding='utf-8'
)

if __name__ == '__main__':
    run(generate_4d_dvf)
