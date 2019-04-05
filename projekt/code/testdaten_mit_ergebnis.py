""" Hinfügen die Ergebnisse in Testdaten
Parameter :
    n/a
Eingabe :
    csv datei (delimiter ';')
    header
        id;input1;input2;output1;ouput2
    filename
        datum_number_szenario_anzahl-werte.csv
            Datumsformat jjjjmmtt
            number 01.. sequenzielle Nummer
            szenario nur a,b oder c
            Format für Anzahl: Wert plus führende NUllen z.B. 0050
Ausgabe :
    csv datei (delimiter ';')
    header
        gleich mit der Eingabe
    data
        output1, output2: jeweils '0', '1'
        Ansonsten gleich mit der Eingabe
    filename
        gleich mit der Eingabe
"""
import csv
import os
from pathlib import Path

PATH = os.path.join(Path(__file__).resolve().parents[1], 'test/') # Python 3.4+
# PATH = Path("../test/")

def erzeuge_testdaten_mit_ergebnisse():
    files = os.listdir(PATH)
    for file in files:
        if file.endswith('-werte.csv'):
            data_in = einlesen(file) # data einlesen from file
            data_out = ersetzen(data_in) # data manipulieren
            schreiben(file.replace('werte', 'ergebnis'), data_out) # data schreiben in output filename


def einlesen( file ):
    # data einlesen from file
    try:
        with open(os.path.join(PATH, file)) as f:
            data_in = list(csv.reader(f, delimiter=";"))

        return data_in

    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))

def ersetzen( data ):
    # data manipulieren, set ouput1 = 0, output2 = 1

    data_out = []
    for line in data:
        if line[0] != 'id':
            line[3] = '0'
            line[4] = '1'
        data_out.append(line)

    return data_out

def schreiben( file_output, data_out):
    # data schreiben in output filename

    try:
        with open(PATH + file_output, 'w') as fo:

            for line in data_out:
                if line[0] == 'id':
                    fo.write(';'.join(line))
                else:
                    fo.write('\n'+';'.join(line))

    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))


if __name__ == '__main__':
    erzeuge_testdaten_mit_ergebnisse()
