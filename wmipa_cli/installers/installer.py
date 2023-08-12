import os
from abc import ABCMeta, abstractmethod
from wmipa_cli.log import logger


class Installer(metaclass=ABCMeta):
    def __init__(self, install_path):
        self.install_path = os.path.abspath(install_path)
        self.paths_to_export = []

    def install(self, yes=False):
        if not self.check_environment(yes):
            logger.error("Installation aborted.")
            return
        logger.info(f"Installing {self.get_name()} in {self.install_path}...")
        here = os.path.abspath(os.path.dirname(__file__))
        os.system(f"mkdir -p {self.install_path}")
        os.chdir(self.install_path)
        self.download()
        self.unpack()
        self.build()
        self.add_to_path()
        os.chdir(here)

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def check_environment(self, yes):
        pass

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def unpack(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def add_to_path(self):
        pass
