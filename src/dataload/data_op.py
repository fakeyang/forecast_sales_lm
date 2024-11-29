# @File         :data_op.py
# @DATE         :2024/11/29
# @Author       :caiayng10
# @Description  :

import os
import sys
import re
import logging
import argparse
import numpy as np
import pandas as pd
from typing import List, Any, Dict, Union
from pydantic import BaseModel, Field

class BaseDataOp(BaseModel):
    """
    
    @param:
    @return:
    """

    