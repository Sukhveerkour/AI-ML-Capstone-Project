from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))
runpy.run_path(str(SRC / 'part_d_hub_segmentation.py'), run_name='__main__')
