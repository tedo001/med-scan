import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MEDSCAN_HOME", tempfile.mkdtemp(prefix="medscan-test-"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
