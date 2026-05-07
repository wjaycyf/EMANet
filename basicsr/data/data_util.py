import os


def recursive_glob(rootdir='.', suffix=''):
    """List all files under rootdir (recursively) whose name ends with suffix."""
    return [filename
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]
