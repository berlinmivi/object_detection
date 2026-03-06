import sys
import subprocess

def pip_install(package:str)->None:
    subprocess.check_call([sys.executable, "-m", "pip", "install",package])
