import os

def session_control6():
    dosya_listesi = os.listdir()
    if "Oranges.xlsx"in dosya_listesi:
        return True

    else:
        return False
def session_control7():
    dosya_listesi = os.listdir()
    if "etki_sayac.png" in dosya_listesi:
        return True

    else:
        return False