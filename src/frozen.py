
import omnifig as fig
import random
import networkx as nx
import torch
from torch import nn
from torch import distributions as distrib
import numpy as np
import zlib
from omnibelt import unspecified_argument

from omnilearn import util
from omnilearn import Function

from .methods import Autoencoder


@fig.AutoModifier('freezable')
class Freezable(Autoencoder):
	def __init__(self, A, freeze_encoder=None, freeze_decoder=None, **kwargs):
		
		if freeze_encoder is None:
			freeze_encoder = A.pull('freeze-encoder', False)
		
		if freeze_decoder is None:
			freeze_decoder = A.pull('freeze-decoder', False)

		if freeze_decoder and freeze_encoder:
			print('WARNING: Freezing both encoder and decoder, this is probably not intended')
			
		super().__init__(A, **kwargs)
		
		if freeze_encoder:
			print('Freezing encoder')
			for param in self.encoder.parameters():
				param.requires_grad = False
		
		if freeze_decoder:
			print('Freezing decoder')
			for param in self.decoder.parameters():
				param.requires_grad = False
		
		self.freeze_encoder = freeze_encoder
		self.freeze_decoder = freeze_decoder
		
		