import os

def session_control1():
    dosya_listesi = os.listdir()
    if "graph1.png"in dosya_listesi:
        return True

    else:
        return False

def session_control2():
    dosya_listesi = os.listdir()
    if "graph2.png"in dosya_listesi:
        return True

    else:
        return False

def session_control3():
    dosya_listesi = os.listdir()
    if "paraya_gore_etki_sinif.png" in dosya_listesi:
        return True

    else:
        return False

def session_control4():
    dosya_listesi = os.listdir()
    if "ailevi_duruma_gore_etki_sinif.png"in dosya_listesi:
        return True

    else:
        return False

def session_control5():
    dosya_listesi = os.listdir()
    if "babanın_isine_gore_etki_sinif.png"in dosya_listesi:
        return True

    else:
        return False