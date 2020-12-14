from __future__ import absolute_import, unicode_literals
from celery import Celery

# Imports for config file parsing
import configparser
import os

# Parse config
# Get current working directory where the config file is located
cwd = os.getcwd()
config = configparser.ConfigParser()
# Parse the config file and retrieve all info in a dict
config.read(os.path.join(cwd, 'celery.ini'))

# Returns broker with parameters present in config file
# Default: use redis broker
def getbroker(broker_type="redis"):
    params = {
      "broker_type": broker_type,
      "password": config[broker_type]['broker_password'],
      "hostname": config[broker_type]['broker_hostname'],
      "port": config[broker_type]['broker_port']
    }
    if params['password']:
        broker = "{broker_type}://:{password}@{hostname}:{port}/0".format(**params)
    else:
        broker = "{broker_type}://{hostname}:{port}/0".format(**params)
    return broker

# Retrieve Redis broker with parameters in config file
broker = getbroker()
# Creates a new Celery app with redis as a broker and a backend
app = Celery('ulhpccelery',
             broker=broker,
             backend=broker,
             include=['ulhpccelery.tasks'])

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=3600,
)

if __name__ == '__main__':
    app.start()
