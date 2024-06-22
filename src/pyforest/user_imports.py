import os
import sys
import shutil
import json
import yaml
import pickle
import tarfile
import zipfile
import tempfile
import threading
import subprocess
import multiprocessing
import re
import time
import math
import random
import urllib
import logging
import warnings
import functools
import itertools
import collections

from glob import glob
from time import sleep
from pathlib import Path
from datetime import datetime
from copy import copy, deepcopy
from numbers import Number
from decimal import Decimal
from fractions import Fraction
from functools import reduce, partial
from collections import Counter, OrderedDict, defaultdict, namedtuple
from urllib.request import urlopen

import cupy as cp
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import cv2
import dlib
import imageio.v3 as iio
import scipy.ndimage as ndi
import pydicom
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import kornia as K
import torchio as tio
import lightning as L
import albumentations as A
import dask
import dask.bag as db
import dask.array as da
import dask.dataframe as dd

from loguru import logger
from omegaconf import OmegaConf
from easydict import EasyDict as edict
from parse import parse, search, findall
from tqdm.auto import tqdm, trange
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from fastprogress import master_bar, progress_bar
from IPython.display import display, clear_output, Video, Audio
from PIL import Image
from skimage.color import gray2rgb, rgb2gray
from skimage.feature import hog, canny
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import disk, ball, skeletonize, remove_small_holes, remove_small_objects
from skimage.morphology import dilation, erosion, opening, closing, binary_dilation, binary_erosion, binary_opening, binary_closing
from skimage.segmentation import clear_border, find_boundaries
from torch.optim import SGD, AdamW, Adam
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, datasets
from torchvision.utils import make_grid, save_image
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from transformers import AutoModelForCausalLM, AutoTokenizer
from joblib import Parallel, delayed
from tqdm.contrib.concurrent import thread_map, process_map
