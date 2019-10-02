# trajectoryprediction_studienarbeit

This project contains the material to train, validate and test three
LSTM-related methods; Vanilla-LSTM (V-LSTM), Occupancy-LSTM (O-LSTM)
and Social LSTM (S-LSTM).

---------------------------------------------------------------------

------------------------------FIRST STEP-----------------------------

	Execute the command in make_directories.sh to create the due folders.

--------------------------------SOURCE-------------------------------

	This code is inspired in the code of A. Vemula
	https://github.com/xuerenlv/social-lstm-tf. It has been modified and
	expanded by Julio Barroeta Esteban as part of a Sudienarbeit for the
	Chair of Circuit Design and Networks of TU Dresden.

---------------------------------------------------------------------

-------------------------------DATASETS------------------------------


    This program contains 3 datasets:

    (A) SIMULATION: Simulates a 10mx10m hall with 5 gates, 1 central
        obstacle and 4 different trajectories

    (B) STPET_RESIDENCE: Simulates trajectories captured in the hall
        of the residence from St. Petersburgerstraße 21, 01069, Dresden,
        Germany. Polynomial interpolation of orders 4 to 6 has been
        previously used to generate functions that describe four different
        trajectories.

    (C) TUDRESDEN: It is formed by the measurements used to generate
        the trajectories of the previous dataset. The scenario is thus the
        hall of the residence St. Petersburgerstraße, 21, 01069, Dresden,
        Germany, as well as above.

-----------------------------------------------------------------------

------------------------------FUNCTIONING------------------------------


	To train the NN:
		1. Open VLSTMtrain (VLSTM), OLSTMtrain (OLSTM) or train (S-LSTM)
		2. If wanted, change the arguments or execute it directly.
		3. Execute the file

	To validate the model:
		1. Open "validation"
		2. Set the default technique (argument "--method") to 1 (SLSTM),
		   2 (OLSTM) or 3 (VLSTM), or introduce it manually
		3. Execute the file

	To test the model:
		1. Open "test"
		2. Set the default technique (argument "--method") to 1 (SLSTM),
		   2 (OLSTM) or 3 (VLSTM), or introduce it manually
		3. Execute the file

	By default, the program trains and tests all the datasets. Usually, not all
	of them are wanted. Any of them can be excluded:

	    - FOR NOT TRAINING with SIMULATION:      Comment line 28 of file utils.py
	    - FOR NOT TRAINING with STPET_RESIDENCE: Comment line 29 of file utils.py
	    - FOR NOT TRAINING with TUDRESDEN:       Comment line 30 of file utils.py

	    - FOR NOT TESTING SIMULATION:            Comment line 35 of file utils.py
	    - FOR NOT TESTING STPET_RESIDENCE:       Comment line 36 of file utils.py
	    - FOR NOT TESTING TUDRESDEN:             Comment line 37 of file utils.py

------------------------------------------------------------------

------------------GENERATION OF THE TRAJECTORIES------------------

	To generate new content in the datasets
	    - For SIMULATION: Execute "simulationgeneration.py"
	    - For STPET_RESIDENCE: Execute "residencesimulationgenerator.py"
	    - For TUDRESDEN: Execute "generationrealtest.py"

-------------------------------------------------------------------
