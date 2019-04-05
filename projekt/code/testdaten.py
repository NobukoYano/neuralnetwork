""" Erzeugen Testdaten
Eingabe :
    Eingabe erfolgt durch console mit Leerzeichen getrennt
    Szenario : a(1 - 100), b(101-1000), c(1001-10000)
    Anzahl : Anzahl der Daten 1 bis 999
Ausgabe :
    csv datei (delimiter ';')
    header
        id;input1;input2;output1;ouput2
    data
        id: aufsteigende sequenzielle Nummer
        input1, input2: Zufallzahl je nach Szenario
        output1, output2: leer
    filename
        datum_number_szenario_anzahl-werte.csv
            Datumsformat jjjjmmtt
            number 01.. sequenzielle Nummer
            szenario nur a,b oder c
            Format für Anzahl: Wert plus führende NUllen z.B. 0050
"""
import os
import time
from pathlib import Path
from random import randint


def erzeuge_testdaten_input():
    szenario, anzahl = input('Bitte geben Sie Szenario(a, b oder c) und Anzahl der Daten(max.999) ein: ').split()
    i = 1  # Sequenzielle Nummer für filename

    # check file name and set number
    while True:
        file_output = time.strftime('%Y%m%d_') + '{0:02d}'.format(i) + '_' + szenario + '_' + '{0:04d}'. \
            format(int(anzahl)) + '-werte.csv'

        if Path("../test/" + file_output).is_file():
            i = i + 1
        else:
            break

    header = 'id;input1;input2;output1;output2'

    # open file and write daten
    try:
        with open(Path(os.path.join(Path(__file__).resolve().parents[1], 'test/'), file_output), 'w') as fo:
        # with open(os.path.join(os.pardir, file_output), 'w') as fo:
            # Write header line
            fo.write(header)

            # write anzahl der Daten je nach Szenarios
            for x in range(int(anzahl)):
                ausgabe = []
                if szenario == 'a':
                    ausgabe = [str(x + 1), str(randint(1, 100)), str(randint(1, 100)), '', '']
                elif szenario == 'b':
                    ausgabe = [str(x + 1), str(randint(101, 1000)), str(randint(101, 1000)), '', '']
                elif szenario == 'c':
                    ausgabe = [str(x + 1), str(randint(1001, 10000)), str(randint(1001, 10000)), '', '']
                else:
                    pass

                fo.write('\n'+';'.join(ausgabe))

    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))


if __name__ == '__main__':
    erzeuge_testdaten_input()
