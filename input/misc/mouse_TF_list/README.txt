Source: http://bioinfo.life.hust.edu.cn/AnimalTFDB/#!/download  
Download date: Dec 29, 2018

Mus_musculus_TF_union.txt
	- join both raw files (TF + co-factors)

entrez_id_mouse_TF.csv
	- use pandas to make symbol + entrez_id csv file
	- fill in missing entrez entries
		- autofill still left 25 missing
		- manually add them: entrez_id_mouse_TF.csv
	- this is generic gene symbol -> [entrez_id list] csv file now

Number of missing entrez IDs in the mouse TF list: 103
Auto fill:
	25 had 0 hits
	1 had 2 hits

18 were manually filled in, 7 were left with 0 hits
	CT573016.1 (rename to Gm49345)
	Gm21060
	LO018689.4
	LO018689.1
	Gm28360
	Gm20422
	Gm29106
