import subprocess
import importlib
import sys

def ensure_dependencies(module_map):
    """
    Ensures that a dictionary of Python modules are installed, prioritizing pygame-ce.
    If pygame-ce is not found, it attempts to install it, and uninstalls regular pygame.

    Args:
        module_map: A dictionary of module names to install and import.
    """
    missing_modules = {}
    for install_name, import_name in module_map.items():
        try:
            importlib.import_module(import_name)
            print(f"{import_name} is already installed.")
        except ImportError:
            missing_modules[install_name] = import_name

    if missing_modules:
        print(f"Installing missing modules: {', '.join(missing_modules.keys())}")
        try:
            # Prioritize pygame-ce
            if 'pygame-ce' in missing_modules:
                try:
                    subprocess.check_call(['python', '-m', 'pip', 'uninstall', '-y', 'pygame'])
                except subprocess.CalledProcessError:
                    print("pygame uninstall failed or was not installed.")
                subprocess.check_call(['python', '-m', 'pip', 'install', 'pygame-ce'])
                del missing_modules['pygame-ce']

            # Install other modules if any
            if missing_modules:
                subprocess.check_call(['python', '-m', 'pip', 'install', *missing_modules.keys()])
            print("Modules installed successfully.")

            # Attempt to import again after installation
            for install_name, import_name in module_map.items():
                try:
                    importlib.import_module(import_name)
                    print(f"{import_name} imported successfully after installation.")
                except ImportError as e:
                    print(f"Failed to import {import_name} after installation: {e}")
                    raise ImportError(f"Failed to import {import_name} after installation.")

        except subprocess.CalledProcessError as e:
            print(f"Failed to install modules: {e}")
            raise ImportError(f"Failed to install modules: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during installation: {e}")
            raise ImportError(f"An unexpected error occurred during installation: {e}")
    else:
        print("All dependencies are already installed.")

if __name__ == "__main__":
    # Example usage:
    try:
        ensure_dependencies({'pygame-ce': 'pygame', 'requests': 'requests'})
        import pygame
        import requests
        print("All imports succeeded. Game can continue.")
    except ImportError as e:
        print(f"Critical import error: {e}. Game cannot continue.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")