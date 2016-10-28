# weatherPrediction
A data.csv-ben találhatóak a felhasznált adatok, dátum szerint csökkenő sorrendben. A "holnapi" jósolt érték az adatok között elsőként szereplő dátumot követő nap. Tehát ha október 31-re szeretnénk a hőmérsékletet megkapni, akkor egészen október 30-ig be kell írni az adatokat.
Az adatok forrása a http://idojarasbudapest.hu/archivalt-idojaras weboldal, ahol átlagoltam a minimum és maximum hőmérsékletet.

Az előrejelzéshez a predict.py futtatandó, aminek a data.csv, scale.pkl és weather.h5 fileokra lesz szüksége.
