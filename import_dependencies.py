import subprocess
import importlib

def ensure_dependencies(module_names):
    """
    Ensures that a list of Python modules are installed. If a module is not found,
    it attempts to install it using pip.

    Args:
        module_names: A list of strings, where each string is a module name.
    """
    missing_modules = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_modules.append(module_name)

    if missing_modules:
        print(f"Installing missing modules: {', '.join(missing_modules)}")
        try:
            subprocess.check_call(['python', '-m', 'pip', 'install', *missing_modules])
            print("Modules installed successfully.")

            # Attempt to import again after installation
            for module_name in missing_modules:
                try:
                    importlib.import_module(module_name)
                    print(f"{module_name} imported successfully after installation.")
                except ImportError as e:
                    print(f"Failed to import {module_name} after installation: {e}")
                    raise ImportError(f"Failed to import {module_name} after installation.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install modules: {e}")
            raise ImportError(f"Failed to install modules: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during installation: {e}")
            raise ImportError(f"An unexpected error occurred during installation: {e}")
    else:
        print("All required modules are already installed.")

if __name__ == "__main__":
    # Example usage:
    try:
        ensure_dependencies(['pygame', 'requests']) #Example modules.
        import pygame
        import requests
        print("All imports succeeded. Game can continue.")
    except ImportError as e:
        print(f"Critical import error: {e}. Game cannot continue.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")