import importlib
import pkgutil
import os

def recursive_find_python_class(target_module: str, class_name: str, current_module: str):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([target_module]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([target_module]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(os.path.join(target_module, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr