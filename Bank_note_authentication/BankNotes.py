# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:35:55 2022

@author: aa
"""

from pydantic import BaseModel

class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float