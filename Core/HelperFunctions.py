class _PrintRep:
    lastPrint = ""


def printRep(text=None):
    if text is None:
        _PrintRep.lastPrint = ""
        print(" ")
    else:
        print("\b" * len(_PrintRep.lastPrint) + text, end='', flush=True)
        _PrintRep.lastPrint = text
