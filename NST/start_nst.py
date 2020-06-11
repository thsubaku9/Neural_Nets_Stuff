import os
from argparse import ArgumentParser

def implementParser():
    parser = ArgumentParser(description = "Neural Style Transfer") 
    parser.add_argument('--content',
        dest = 'content', help = 'content image', metavar = 'CONTENT',required = True, nargs = 1)
    parser.add_argument('--style',
        dest = 'style', help = 'style image', metavar = 'STYLE', required = True, nargs = 1)
    
#we need to figure out what all we wish to support and what all we don't. DO THAT !
