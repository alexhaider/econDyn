#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 20:58:56 2025

@author: alex
"""

class Ds:
    """
    class to simulate a generic dynamical system with state x and map h
    """
    
    def __init__(self, h, x):
        self.h = h
        self.x = x
        
    def update(self):
        self.x = self.h(self.x)
        
    def trajectory(self, n: int):
        traj = []
        for i in range(n):
            traj.append(self.x)
            self.update()
        return traj
            