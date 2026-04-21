import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/david/Desktop/laiting/navigation_codex_legacy_baseline/install/mapless_nav2'
