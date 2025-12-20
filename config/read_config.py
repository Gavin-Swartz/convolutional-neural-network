import yaml

config_path = './config/config.yaml'

def read_config_file():
  with open(config_path) as file:
    try:
      return yaml.safe_load(file)
    except yaml.YAMLError as e:
      print(e.with_traceback())
    