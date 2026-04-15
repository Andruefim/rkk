import cProfile
import pstats
import urllib.request

try:
    print("Triggering single tick by calling simulation api...")
    # Actually simulation runs in background loop perhaps?
    # Let's write a small profiler that attaches to the loop or just profiles the process if possible.
    # A simpler way is to fetch some stats if there are any, or run a few ticks directly if we can import the singleton
except Exception as e:
    pass
