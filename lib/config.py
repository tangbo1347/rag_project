import configparser





def config_get_item(section, item):
    config = configparser.ConfigParser()

    config.read("config.ini", encoding="utf-8")

    return config.get(section, item)