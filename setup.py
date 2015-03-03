try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Anchor Graph',
    'author': 'Paul Jacobs',
    'url': '',
    'download_url': '',
    'author_email': '',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['anchor_graph'],
    'scripts': [],
    'name': 'anchor_graph'
}

setup(**config)
