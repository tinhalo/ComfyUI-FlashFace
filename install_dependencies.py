import subprocess
import sys

def install_package(package):
    print(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}. Error: {e}")
        return False
    return True

if __name__ == "__main__":
    # Install pydash
    if install_package("pydash==7.0.7"):
        print("All dependencies installed successfully.")
    else:
        print("Failed to install some dependencies. Please check the logs above.")