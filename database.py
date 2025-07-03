# database.py
import os
import configparser
import MySQLdb as mysql

def GetConnection(section='hess', cfg_path=None):
    """
    Read your ~/.dbtoolsrc (or another path) and return a MySQLdb connection.
    section: one of 'hess', 'astro', or 'test'
    """
    if cfg_path is None:
        cfg_path = os.path.expanduser('~/.dbtoolsrc')
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    if section not in cfg:
        raise ValueError(f"Section {section!r} not found in {cfg_path}")
    
    params = cfg[section]
    return mysql.connect(
        host=params.get('host'),
        user=params.get('user'),
        passwd=params.get('password'),
        db=params.get('database'),
    )
