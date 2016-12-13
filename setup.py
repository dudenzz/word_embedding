from setuptools import setup, find_packages
from codecs import open
from os import path


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
	name = "Word Embedding Tools", 
	version = '1.0.0', 
	description = "This is a set of tools for Word Embedding capable of executing semantic tasks", 
	url = "https://github.com/dudenzz/word_embedding",
	author = "Jakub Dutkiewicz",
	author_email = "jakub.dutkiewicz@put.poznan.pl",
	license = "MIT",
	classifiers = [
		'Development Status :: 2',
		'Intended Audience :: Computational Linguistics Scientists',
		'License :: MIT',
	 	'Programming Language :: Python :: 2.7'
	],
	keywords = 'glove hubness preprocessing similarity question answering computational linguistics',
	packages = find_packages(exclude=['contrib','docs','tests'])
	)
