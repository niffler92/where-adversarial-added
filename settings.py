import os
from pathlib import Path
from nsml import NSML_NFS_OUTPUT

if NSML_NFS_OUTPUT:
    PROJECT_ROOT = Path(os.path.join(NSML_NFS_OUTPUT, 'kyhoon', 'ACE-Defense'))
else:
    PROJECT_ROOT = Path(__file__).resolve().parent
